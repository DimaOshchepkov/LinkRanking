#include <benchmark/benchmark.h>
#include <string>
#include <vector>
#include "..\LinkRanking\LinkRanking.h"
#include <format>

#define STRINGIFY(x) #x

#define EXPAND(x) STRINGIFY(x)

using namespace std;
string solutionDir;
string getSolutionDir(std::string short_path)
{
    if (!solutionDir.empty())
        return solutionDir + "\\" + short_path;
    string s = EXPAND(SOLDIR);
    s.erase(0, 1);         // erase the first quote
    s.erase(s.size() - 2); // erase the last quote and the dot

    size_t found = s.find_last_of('\\');

    if (found != std::string::npos)
    {
        s = s.substr(0, found);
    }

    solutionDir = s;
    return solutionDir + "\\" + short_path;
}

static void BM_build_transition(benchmark::State &state)
{
    using namespace ranking;

    auto path = getSolutionDir(R"(res\rating.csv)");
    for (auto _ : state)
    {
        auto x = TransitionMatrixBuilder()
                     .SetPath(path)
                     .SetSourceName("userId")
                     .SetTargetName("movieId")
                     .SetEdgeAttributeName("rating")
                     .Build();
        benchmark::DoNotOptimize(x);
    }
}

static void BM_Pagerank_MovieLens(benchmark::State &state, double sample)
{
    using namespace ranking;
    auto path = getSolutionDir(R"(res\rating.csv)");
    auto [M, map] = TransitionMatrixBuilder()
                        .SetPath(path)
                        .SetSourceName("userId")
                        .SetTargetName("movieId")
                        .SetEdgeAttributeName("rating")
                        .SetSample(sample)
                        .Build();

    int row_index = 1;
    auto dense_vec = get_personalization_from_matrix_row(M, row_index);

    for (auto _ : state)
    {
        auto pagerank = PageRankFactory()
                            .SetTransitionMatrix(M)
                            .SetPersonalizationVector(dense_vec)
                            .Build();
        benchmark::DoNotOptimize(pagerank);
    }
}

static void BM_Hits_MovieLens(benchmark::State &state, double sample)
{
    using namespace ranking;
    auto path = getSolutionDir(R"(res\rating.csv)");
    auto [M, map] = TransitionMatrixBuilder()
                        .SetPath(path)
                        .SetSourceName("userId")
                        .SetTargetName("movieId")
                        .SetEdgeAttributeName("rating")
                        .SetSample(sample)
                        .Build();

    int row_index = 1;
    auto dense_vec = get_personalization_from_matrix_row(M, row_index);

    for (auto _ : state)
    {
        auto pagerank = HITSFactory()
                            .SetTransitionMatrix(M)
                            .Build();
        benchmark::DoNotOptimize(pagerank);
    }
}

int main(int argc, char **argv)
{
    for (double sample = 0.1; sample <= 1.01; sample += 0.1)
    {
        benchmark::RegisterBenchmark(
            std::format("BM_Pagerank_MovieLens/sample-{:.2f}", sample),
            BM_Pagerank_MovieLens, sample)
            ->Iterations(3)
            ->Unit(benchmark::kSecond);

        benchmark::RegisterBenchmark(
            std::format("BM_Hits_MovieLens/sample-{:.2f}", sample),
            BM_Hits_MovieLens, sample)
            ->Iterations(3)
            ->Unit(benchmark::kSecond);
    }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}