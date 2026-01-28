## Теорема о неявной функции

Рассмотрим функцию $F: \mathbb{R}^{n+m} \to \mathbb{R}^m$, заданную как $F(\mathbf{x}, \mathbf{y}) = \mathbf{0}$, где $\mathbf{x} \in \mathbb{R}^n$ и $\mathbf{y} \in \mathbb{R}^m$. Если в некоторой точке $(\mathbf{a}, \mathbf{b})$ выполняется $F(\mathbf{a}, \mathbf{b}) = \mathbf{0}$, и частная матрица Якоби по $\mathbf{y}$ в этой точке, $\frac{\partial F}{\partial \mathbf{y}}(\mathbf{a}, \mathbf{b})$, является невырожденной (т.е. её определитель отличен от нуля), то существует открытое окрестность $U$ точки $\mathbf{a}$ и открытая окрестность $V$ точки $\mathbf{b}$, а также единственная непрерывно дифференцируемая функция $G: U \to V$ такая, что $\mathbf{y} = G(\mathbf{x})$ для всех $\mathbf{x} \in U$, и $F(\mathbf{x}, G(\mathbf{x})) = \mathbf{0}$.

Более того, матрица Якоби функции $G$ в точке $\mathbf{a}$ задается следующей формулой:

$$ \frac{\partial G}{\partial \mathbf{x}}(\mathbf{a}) = - \left( \frac{\partial F}{\partial \mathbf{y}}(\mathbf{a}, \mathbf{b}) \right)^{-1} \frac{\partial F}{\partial \mathbf{x}}(\mathbf{a}, \mathbf{b}) $$