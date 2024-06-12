#pragma once
#include "pch.h"
#include <rapidcsv.h>
namespace ranking
{
    enum Type {
        SOURCE,
        TARGET,
    };


class TransitionMatrixBuilder {
    public:
        TransitionMatrixBuilder() = default;

        TransitionMatrixBuilder& SetPath(const std::string& path);
        TransitionMatrixBuilder& SetSourceName(const std::string& source_name);
        TransitionMatrixBuilder& SetTargetName(const std::string& target_name);
        TransitionMatrixBuilder& SetEdgeAttributeName(const std::string& edge_attr_name);
        TransitionMatrixBuilder& SetDirectional(bool directional);
        TransitionMatrixBuilder& SetLabelParams(const rapidcsv::LabelParams& label_params);
        TransitionMatrixBuilder& SetSample(double sample);
        TransitionMatrixBuilder& SetSeparator(rapidcsv::SeparatorParams sep);


        std::tuple<Eigen::SparseMatrix<double>, std::vector<std::pair<std::string, Type>>>
            Build() const;

    private:
        std::string path_ = "";
        std::string source_name_ = "source";
        std::string target_name_ = "target";
        std::string edge_attr_name_ = "edge_attr";
        bool directional_ = false;
        rapidcsv::LabelParams label_params_ = rapidcsv::LabelParams(0, -1);
        double sample = 1;
        rapidcsv::SeparatorParams sep = rapidcsv::SeparatorParams(',');
    };

class TransitionMatrixFromTxtBuilder {
public:
    TransitionMatrixFromTxtBuilder() = default;

    TransitionMatrixFromTxtBuilder& SetPath(const std::string& path);
    TransitionMatrixFromTxtBuilder& SetSourceName(int source_name);
    TransitionMatrixFromTxtBuilder& SetTargetName(int target_name);
    TransitionMatrixFromTxtBuilder& SetEdgeAttributeName(int edge_attr_name);
    TransitionMatrixFromTxtBuilder& SetDirectional(bool directional);
    TransitionMatrixFromTxtBuilder& SetLabelParams(const rapidcsv::LabelParams& label_params);
    TransitionMatrixFromTxtBuilder& SetSample(double sample);
    TransitionMatrixFromTxtBuilder& SetSeparator(rapidcsv::SeparatorParams sep);


    std::tuple<Eigen::SparseMatrix<double>, std::vector<std::pair<std::string, Type>>>
        Build() const;

private:
    std::string path_ = "";
    int source = 0;
    int target = 1;
    int edge = -1;
    bool directional_ = false;
    rapidcsv::LabelParams label_params_ = rapidcsv::LabelParams(-1, -1);
    double sample = 1;
    rapidcsv::SeparatorParams sep = rapidcsv::SeparatorParams('\t');
};

Eigen::VectorXd get_pagerank(const Eigen::SparseMatrix<double>& trasition,
    const Eigen::VectorXd& person_vec,
    double a, double b, double c, int max_iter);


class PageRankFactory {
public:
    PageRankFactory() = default;

    PageRankFactory& SetTransitionMatrix(const  Eigen::SparseMatrix<double>& matrix);
    PageRankFactory& SetPersonalizationVector(const Eigen::VectorXd& vector);
    PageRankFactory& SetAlpha(double alpha);
    PageRankFactory& SetBeta(double beta);
    PageRankFactory& SetGamma(double gamma);
    PageRankFactory& SetMaxIterations(int max_iterations);

    Eigen::VectorXd Build() const;

private:
    const Eigen::SparseMatrix<double>* transition_matrix_;
    const Eigen::VectorXd* personalization_vector_;
    double alpha_ = 0.80;
    double beta_ = 0.15;
    double gamma_ = 0.05;
    int max_iterations_ = 100;
};

Eigen::VectorXd get_personalization_from_matrix_row(
    const Eigen::SparseMatrix<double>& matrix,
    int index);


class HITSFactory {
public:
    HITSFactory() = default;

    HITSFactory& SetTransitionMatrix(const Eigen::SparseMatrix<double>& matrix);
    HITSFactory& SetMaxIter(int iterations);

    std::tuple<Eigen::VectorXd, Eigen::VectorXd> Build() const;

private:
    const Eigen::SparseMatrix<double>* trasition;
    int max_iter = 100;
};
}
