# Курсовой проект на тему "Разработка системы ссылочного ранжирования"

Этот курсовой проект был создан в рамках дисциплины "Терория алгоритмов". В нем рассматриваются 2 наиболее распространенных алгоритма ссылочного ранжирования: Pagrank и HITS.

## Быстрый старт
Пример использования pagrank :
```cpp
#include "..\LinkRanking\LinkRanking.h"

auto x = TransitionMatrixBuilder()
            .SetPath("path/to/datafram")
            .SetSourceName("userId")
            .SetTargetName("movieId")
            .SetEdgeAttributeName("rating")
            .Build();

int row_index = 1;
auto dense_vec = get_personalization_from_matrix_row(M,row_index);


auto pagerank = PageRankFactory()
    .SetTransitionMatrix(M)
    .SetPersonalizationVector(dense_vec)
    .Build();
```

pagerank - будет вектором персонализации.

Пример использования HITS:
```cpp
#include "..\LinkRanking\LinkRanking.h"

auto x = TransitionMatrixBuilder()
            .SetPath("path/to/datafram")
            .SetSourceName("userId")
            .SetTargetName("movieId")
            .SetEdgeAttributeName("rating")
            .Build();

int row_index = 1;
auto dense_vec = get_personalization_from_matrix_row(M,row_index);


auto [auth, hubs] = HITSFactory()
    .SetTransitionMatrix(M)
    .SetPersonalizationVector(dense_vec)
    .Build();

```

Этот проект является вспомогательным (здесь только замеры времени). Основное исследование [тут](https://github.com/DimaOshchepkov/PageRank-Course-Project/blob/main/plot_comparision.ipynb).