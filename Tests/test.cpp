#include "pch.h"

#include "..\LinkRanking\LinkRanking.h"

#define STRINGIFY(x) #x

#define EXPAND(x) STRINGIFY(x)

using namespace std;
string solutionDir;
string getSolutionDir(std::string short_path) {
    if (!solutionDir.empty())
        return solutionDir + "\\" + short_path;
    string s = EXPAND(SOLDIR);
    s.erase(0, 1); // erase the first quote
    s.erase(s.size() - 2); // erase the last quote and the dot

    size_t found = s.find_last_of('\\');

    if (found != std::string::npos) {
        s = s.substr(0, found);
    }

    solutionDir = s;
    return solutionDir + "\\" + short_path;
}

TEST(TransitionMatrixBuilder, standart_case) {
	using namespace ranking;
    auto path = getSolutionDir("Tests\\res\\rating.csv");
    auto [M, map] = TransitionMatrixBuilder()
        .SetPath(path)
        .SetSourceName("userId")
        .SetTargetName("movieId")
        .SetEdgeAttributeName("rating")
        .Build();

    auto [M2, map2] = TransitionMatrixBuilder()
        .SetPath(path)
        .SetSourceName("userId")
        .SetTargetName("movieId")
        .SetEdgeAttributeName("rating")
        .Build();
    EXPECT_EQ(M.nonZeros(), 3*2);
}

TEST(get_personalization_from_matrix_row, standart_case) {
    using namespace ranking;
    auto path = getSolutionDir("Tests\\res\\rating.csv");
    auto [M, map] = TransitionMatrixBuilder()
        .SetPath(path)
        .SetSourceName("userId")
        .SetTargetName("movieId")
        .SetEdgeAttributeName("rating")
        .Build();

    int row_index = 1;
    auto dense_vec = get_personalization_from_matrix_row(M, row_index);
    EXPECT_EQ(dense_vec.size(), 4);
}

TEST(HITSFactory, standart_case) {
    using namespace ranking;
    auto path = getSolutionDir("Tests\\res\\rating.csv");
    auto [M, map] = TransitionMatrixBuilder()
        .SetPath(path)
        .SetSourceName("userId")
        .SetTargetName("movieId")
        .SetEdgeAttributeName("rating")
        .Build();

    int row_index = 1;
    auto dense_vec = get_personalization_from_matrix_row(M, row_index);

    auto [auth, hub] = HITSFactory()
            .SetTransitionMatrix(M)
            .Build();
    
    EXPECT_EQ(auth.size(), 4);
}


TEST(TransitionBuilder, SetSample) {
    using namespace ranking;
    auto path = getSolutionDir("Tests\\res\\rating.csv");
    auto [M, map] = TransitionMatrixBuilder()
        .SetPath(path)
        .SetSourceName("userId")
        .SetTargetName("movieId")
        .SetEdgeAttributeName("rating")
        .SetSample(0.5)
        .Build();

    auto [M2, map2] = TransitionMatrixBuilder()
        .SetPath(path)
        .SetSourceName("userId")
        .SetTargetName("movieId")
        .SetEdgeAttributeName("rating")
        .SetSample(1)
        .Build();

    EXPECT_TRUE(M2.rows() > M.rows());
}