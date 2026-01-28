CREATE SCHEMA IF NOT EXISTS rag;

-- ARTICLES
CREATE TABLE IF NOT EXISTS rag.articles (
  article_id      INT PRIMARY KEY,
  source_url      TEXT NOT NULL,
  title           TEXT,
  content         TEXT,
  revision        INT,
  article_hash    TEXT NOT NULL,          -- e.g. sha256 hex
  updated_at_src  TEXT,
  created_at      TIMESTAMPTZ DEFAULT now(),
  updated_at      TIMESTAMPTZ DEFAULT now()
);

-- CHUNKS
CREATE TABLE IF NOT EXISTS rag.chunks (
  chunk_id      TEXT PRIMARY KEY,   -- "article_id-chunk_ord", "000647-0017", "002142-0003"
  article_id    INT NOT NULL REFERENCES rag.articles(article_id) ON DELETE CASCADE,
  chunk_ord     INT NOT NULL CHECK (chunk_ord > 0),
  section_path  JSONB NOT NULL DEFAULT '[]'::jsonb,
  block_types   JSONB NOT NULL DEFAULT '[]'::jsonb,
  text          TEXT NOT NULL,
  char_len      INT NOT NULL,
  chunk_hash    TEXT NOT NULL,
  created_at    TIMESTAMPTZ DEFAULT now(),
  updated_at    TIMESTAMPTZ DEFAULT now(),
  UNIQUE (article_id, chunk_ord)
);

CREATE INDEX IF NOT EXISTS idx_chunks_article_id ON rag.chunks(article_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section_path_gin ON rag.chunks USING GIN(section_path);
CREATE INDEX IF NOT EXISTS idx_chunks_block_types_gin ON rag.chunks USING GIN(block_types);

-- IMAGES
CREATE TABLE IF NOT EXISTS rag.images (
  image_id    BIGSERIAL PRIMARY KEY,
  local_path  TEXT NOT NULL UNIQUE,
  alt         TEXT,
  caption     TEXT
);

-- CHUNK <-> IMAGES with order
CREATE TABLE IF NOT EXISTS rag.chunk_images (
  chunk_id  TEXT NOT NULL REFERENCES rag.chunks(chunk_id) ON DELETE CASCADE,
  image_id  BIGINT NOT NULL REFERENCES rag.images(image_id) ON DELETE CASCADE,
  ord       INT NOT NULL,
  PRIMARY KEY (chunk_id, ord),
  UNIQUE (chunk_id, image_id)
);

CREATE INDEX IF NOT EXISTS idx_chunk_images_image_id ON rag.chunk_images(image_id);
