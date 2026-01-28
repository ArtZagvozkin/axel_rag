Создание чат-бота на Perl для Telegram с использованием Gemini API и принципов Чистой Архитектуры (Clean Architecture) — это довольно сложная, но выполнимая задача.

Поскольку Perl не имеет такого обилия готовых асинхронных фреймворков, как Python (например, `aiohttp` или `FastAPI`), мы будем использовать стандартные модули для HTTP-запросов (`LWP::UserAgent`) и асинхронного цикла (хотя для простоты в этом примере мы сделаем его синхронным, который опрашивает Telegram каждые N секунд).

Мы структурируем код по слоям Чистой Архитектуры:

1. **Entities (Сущности):** Определяют структуру данных (Сообщение, Контекст).
2. **Use Cases (Варианты Использования):** Содержат бизнес-логику (например, `ProcessMessage`, `ManageContext`).
3. **Adapters (Адаптеры):** Внешние интерфейсы (Telegram API, Gemini API, Логгер, Хранилище).
4. **Presentation (Презентация):** Главный цикл бота, который вызывает Use Cases.

### Требования и Подготовка

1. **Perl:** Установлен.
2. **Модули:** Вам понадобятся:
   * `LWP::UserAgent` (для HTTP-запросов)
   * `JSON` (для работы с JSON)
   * `Time::HiRes` (для задержек)
   * `DateTime` (для логирования)

Установите их через CPAN:
```bash
cpan LWP::UserAgent JSON Time::HiRes DateTime
```

3. **Ключи:**
   * `TELEGRAM_BOT_TOKEN`: Ваш токен от BotFather.
   * `GEMINI_API_KEY`: Ваш ключ для Gemini API.

---

## Структура Проекта

Создадим директорию `perl_gemini_bot/` и в ней следующие файлы:

```
perl_gemini_bot/
├── lib/
│   ├── Entities.pm        # Сущности
│   ├── UseCases.pm        # Бизнес-логика
│   └── Adapters/
│       ├── GeminiAdapter.pm # Адаптер для Gemini
│       ├── TelegramAdapter.pm # Адаптер для Telegram
│       └── FileLogger.pm    # Адаптер для логирования
└── bot.pl                 # Главный исполняемый файл (Presentation)
```

### 1. Entities.pm (Сущности)

Определяет структуру данных.

```perl
# lib/Entities.pm
package Entities;
use strict;
use warnings;
use constant MAX_CONTEXT_SIZE => 10;

# Хранилище контекста для каждого чата
my %chat_context;

sub add_to_context {
    my ($chat_id, $role, $content) = @_;

    # Инициализация, если чат новый
    $chat_context{$chat_id} ||= [];

    push @{$chat_context{$chat_id}}, {
        role => $role,
        parts => [ { text => $content } ]
    };

    # Обрезка контекста до последних N сообщений
    if (@{$chat_context{$chat_id}} > MAX_CONTEXT_SIZE) {
        @{$chat_context{$chat_id}} = splice(@{$chat_context{$chat_id}}, -MAX_CONTEXT_SIZE);
    }
}

sub get_context {
    my ($chat_id) = @_;
    return $chat_context{$chat_id} || [];
}

sub reset_context {
    my ($chat_id) = @_;
    delete $chat_context{$chat_id};
    return 1;
}

sub get_max_context_size {
    return MAX_CONTEXT_SIZE;
}

1;
```

### 2. Adapters

#### 2.1. Adapters/FileLogger.pm (Логгер)

```perl
# lib/Adapters/FileLogger.pm
package Adapters::FileLogger;
use strict;
use warnings;
use DateTime;

my $LOG_FILE = 'bot.log';

sub log {
    my ($level, $message) = @_;
    my $dt = DateTime->now->ymd('%Y-%m-%d %H:%M:%S');
    
    open(my $fh, '>>', $LOG_FILE) or die "Не удалось открыть лог-файл: $!";
    print $fh "[$dt] [$level] $message\n";
    close $fh;
}

sub info {
    my ($msg) = @_;
    log('INFO', $msg);
}

sub error {
    my ($msg) = @_;
    log('ERROR', $msg);
}

1;
```

#### 2.2. Adapters/TelegramAdapter.pm (Telegram API)

```perl
# lib/Adapters/TelegramAdapter.pm
package Adapters::TelegramAdapter;
use strict;
use warnings;
use LWP::UserAgent;
use JSON;

sub new {
    my ($class, $token) = @_;
    my $self = {
        token => $token,
        ua    => LWP::UserAgent->new(timeout => 30),
        api_url => "https://api.telegram.org/bot$token/",
        json  => JSON->new->allow_nonref,
    };
    return bless $self, $class;
}

sub _call_api {
    my ($self, $method, $params) = @_;
    my $url = $self->{api_url} . $method;

    my $response = $self->{ua}->post($url, Content => $self->{json}->encode($params),
                                       'Content-Type' => 'application/json');

    if (!$response->is_success) {
        warn "Telegram API Error: " . $response->status_line;
        return undef;
    }

    return $self->{json}->decode($response->content);
}

sub get_updates {
    my ($self, $offset) = @_;
    my $params = { timeout => 20 }; # Long Polling
    $params->{offset} = $offset if $offset;
    
    my $data = $self->_call_api('getUpdates', $params);
    return $data && $data->{ok} ? $data->{result} : [];
}

sub send_message {
    my ($self, $chat_id, $text) = @_;
    $self->_call_api('sendMessage', {
        chat_id => $chat_id,
        text    => $text,
    });
}

1;
```

#### 2.3. Adapters/GeminiAdapter.pm (Gemini API)

```perl
# lib/Adapters/GeminiAdapter.pm
package Adapters::GeminiAdapter;
use strict;
use warnings;
use LWP::UserAgent;
use JSON;

sub new {
    my ($class, $api_key) = @_;
    my $self = {
        api_key => $api_key,
        ua      => LWP::UserAgent->new(timeout => 60), # Увеличиваем таймаут для LLM
        api_url => "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=$api_key",
        json    => JSON->new->allow_nonref,
    };
    return bless $self, $class;
};

sub generate_content {
    my ($self, $messages) = @_;
    
    # Форматирование контекста в формат Gemini
    my $payload = {
        contents => $messages,
    };

    my $response = $self->{ua}->post($self->{api_url},
        Content => $self->{json}->encode($payload),
        'Content-Type' => 'application/json'
    );

    if (!$response->is_success) {
        warn "Gemini API Error: " . $response->status_line . " | " . $response->content;
        return "Извините, произошла ошибка при обращении к Gemini.";
    }
    
    my $data = $self->{json}->decode($response->content);
    
    # Извлечение текста из сложной структуры Gemini
    if (my $text = $data->{candidates}[0]{content}{parts}[0]{text}) {
        return $text;
    }
    
    # Обработка случая, когда ответ заблокирован или пуст
    return "Gemini вернул пустой или заблокированный ответ.";
}

1;
```

### 3. Use Cases (Бизнес-логика)

#### 3.1. UseCases.pm

```perl
# lib/UseCases.pm
package UseCases;
use strict;
use warnings;
use Adapters::GeminiAdapter;
use Entities;
use Adapters::FileLogger;

sub new {
    my ($class, $gemini_adapter, $telegram_adapter) = @_;
    return bless {
        gemini    => $gemini_adapter,
        telegram  => $telegram_adapter,
    }, $class;
}

# Вариант использования: Обработка входящего сообщения
sub HandleIncomingMessage {
    my ($self, $update) = @_;
    
    my $message = $update->{message};
    return unless $message;
    
    my $chat_id = $message->{chat}->{id};
    my $text = $message->{text} || '';
    my $username = $message->{from}->{username} || 'User';

    Adapters::FileLogger->info("Received message from $chat_id ($username): $text");

    my $response_text = "";

    # --- 1. Обработка команд ---
    if ($text =~ m{^/start$}) {
        $response_text = "Привет! Я бот, использующий Gemini API. Можете задавать вопросы. Используйте /reset для сброса контекста.";
        Entities::reset_context($chat_id);
    } 
    elsif ($text =~ m{^/reset$}) {
        Entities::reset_context($chat_id);
        $response_text = "Контекст диалога для чата $chat_id сброшен.";
        Adapters::FileLogger->info("Context reset for $chat_id");
    }
    # --- 2. Обработка обычных сообщений ---
    else {
        # Добавляем сообщение пользователя в контекст (роль: user)
        Entities::add_to_context($chat_id, 'user', $text);
        
        # Получаем текущий контекст
        my @context = Entities::get_context($chat_id);
        
        # Вызываем Gemini
        $response_text = $self->{gemini}->generate_content(\@context);
        
        # Добавляем ответ Gemini в контекст (роль: model)
        Entities::add_to_context($chat_id, 'model', $response_text);
        
        Adapters::FileLogger->info("Generated response for $chat_id.");
    }
    
    # Отправка ответа
    $self->{telegram}->send_message($chat_id, $response_text);
}

1;
```

### 4. Presentation (Главный цикл бота)

#### bot.pl

```perl
#!/usr/bin/perl
use strict;
use warnings;
use LWP::UserAgent;
use JSON;
use Time::HiRes qw(sleep);

use constant {
    TELEGRAM_TOKEN    => $ENV{TELEGRAM_BOT_TOKEN} // die "Не установлен TELEGRAM_BOT_TOKEN",
    GEMINI_API_KEY    => $ENV{GEMINI_API_KEY}     // die "Не установлен GEMINI_API_KEY",
    POLL_INTERVAL_SEC => 1,
};

# Загружаем наши модули
use lib './lib';
use Adapters::TelegramAdapter;
use Adapters::GeminiAdapter;
use Adapters::FileLogger;
use UseCases;
use Entities;

Adapters::FileLogger->info("Bot starting up...");

# --- Инициализация Адаптеров (Dependencies Injection) ---
my $telegram_adapter = Adapters::TelegramAdapter->new(TELEGRAM_TOKEN);
my $gemini_adapter = Adapters::GeminiAdapter->new(GEMINI_API_KEY);
my $use_cases = UseCases->new($gemini_adapter, $telegram_adapter);

my $last_update_id = 0;

sub process_updates {
    my @updates = @_;
    
    foreach my $update (@updates) {
        # Обновляем ID, чтобы не обрабатывать одно и то же сообщение дважды
        $last_update_id = $update->{update_id} + 1;

        # Вызываем Use Case для обработки сообщения
        $use_cases->HandleIncomingMessage($update);
    }
}

# --- Главный цикл ---
while (1) {
    my @updates = $telegram_adapter->get_updates($last_update_id);
    
    if (@updates) {
        process_updates(@updates);
    } else {
        # Если обновлений нет, просто ждем
        Adapters::FileLogger->info("No updates. Sleeping.");
    }
    
    sleep(POLL_INTERVAL_SEC);
}

__END__
```

### Как запустить

1. **Настройте переменные окружения:**
   ```bash
   export TELEGRAM_BOT_TOKEN="ВАШ_ТОКЕН"
   export GEMINI_API_KEY="ВАШ_GEMINI_КЛЮЧ"
   ```

2. **Перейдите в директорию проекта:**
   ```bash
   cd perl_gemini_bot
   ```

3. **Запустите бота:**
   ```bash
   perl bot.pl
   ```

4. **Протестируйте в Telegram:**
   * Отправьте `/start`.
   * Отправьте вопрос, например: "Напиши короткий стих о Perl." (Gemini ответит и запомнит).
   * Отправьте следующий вопрос, зависящий от контекста: "А теперь сделай его в стиле Шекспира."
   * Отправьте `/reset`, затем снова задайте первый вопрос, чтобы убедиться, что контекст сброшен.

### Соответствие Чистой Архитектуре

1. **Entities (Entities.pm):** Содержат только чистые структуры данных (контекст, размер). Они не знают о Telegram или Gemini.
2. **Use Cases (UseCases.pm):** Содержат всю бизнес-логику (`HandleIncomingMessage`). Они зависят от Адаптеров (через внедрение зависимостей в `new`), но Адаптеры не зависят от них.
3. **Adapters (TelegramAdapter.pm, GeminiAdapter.pm, FileLogger.pm):** Это внешние интерфейсы. Они знают, как общаться с внешними системами (HTTP POST, парсинг JSON), и преобразуют внешние данные в формат, понятный Use Cases (и наоборот).
4. **Presentation (bot.pl):** Главный цикл, который управляет потоком и отвечает за инициализацию и вызов Use Cases. Он — самый "грязный" слой, так как отвечает за запуск приложения, но он не содержит бизнес-логики.