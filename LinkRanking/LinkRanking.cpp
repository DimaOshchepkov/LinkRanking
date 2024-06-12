// LinkRanking.cpp : Определяет функции для статической библиотеки.
//

#include "pch.h"
#include "framework.h"

#include "LinkRanking.h"

#include <set>
#include <boost/container/flat_map.hpp>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>
#include <tuple>
#include <rapidcsv.h>
#include <algorithm>
#include <optional>
#include <assert.h>

using namespace std;


namespace ranking
{
    using namespace Eigen;

    void _norm_matrix_rows(SparseMatrix<double>& matrix, const Eigen::VectorXd& values) {
        // Нормализуем строки
        for (int i = 0; i < matrix.outerSize(); ++i) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, i); it; ++it) {
                if (values(it.row()) != 0.0) {
                    it.valueRef() /= values(it.row());
                }
            }
        }
    }

    Eigen::VectorXd _get_sum_matrix_rows(const SparseMatrix<double>& matrix) {
        // Вектор для хранения сумм строк
        Eigen::VectorXd rowSums(matrix.rows());
        rowSums.setZero();

        // Вычисляем сумму элементов в каждой строке
        for (int i = 0; i < matrix.outerSize(); ++i) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, i); it; ++it) {
                rowSums(it.row()) += it.value();
            }
        }
        return rowSums;
    }



    std::tuple<
        boost::unordered_map<std::string, int>,
        boost::unordered_map<std::string, int>,
        std::vector<std::pair<std::string, Type>>>
        _get_mapping(
            const std::vector<std::string>& source,
            const std::vector<std::string>& target) {

        boost::unordered_set<std::string> source_set(source.begin(), source.end());
        boost::unordered_set<std::string> target_set(target.begin(), target.end());

        int index_map = 0;
        boost::unordered_map<std::string, int> object_source_to_integer(source_set.size());
        boost::unordered_map<std::string, int> object_target_to_integer(target_set.size());
        std::vector<std::pair<std::string, Type>> integer_to_object;
        integer_to_object.reserve(source_set.size() + target_set.size());
        for (auto el : source_set) {
            object_source_to_integer.insert({ el, index_map });
            integer_to_object.push_back({ el, Type::SOURCE });
            index_map++;
        }
        for (auto el : target_set) {
            object_target_to_integer.insert({ el, index_map });
            integer_to_object.push_back({ el, Type::SOURCE });
            index_map++;
        }

        return { object_source_to_integer , object_target_to_integer, integer_to_object };
    }

    std::vector<Eigen::Triplet<double>>
        _get_triplets(
            const boost::unordered_map<std::string, int>& object_source_to_integer,
            const boost::unordered_map<std::string, int>& object_target_to_integer,
            const std::vector<std::string>& source,
            const std::vector<std::string>& target,
            const std::vector<double>& edge_attr,
            bool directional)
    {
        int count_row = edge_attr.size();
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(count_row);
        for (int i = 0; i < count_row; i++) {
            auto from = object_source_to_integer.at(source[i]);
            auto to = object_target_to_integer.at(target[i]);
            triplets.emplace_back(from, to, edge_attr[i]);
            if (!directional) {
                triplets.emplace_back(to, from, edge_attr[i]);
            }
        }
        return triplets;
    }

    std::tuple<
        std::vector<std::string>,
        std::vector<std::string>,
        std::vector<double>>
        _extract_column(
            const std::string& path,
            const std::string& source_name,
            const std::string& target_name,
            const std::string& edge_attr_name,
            rapidcsv::LabelParams label_params,
            double sample,
            rapidcsv::SeparatorParams sep)
    {
        try {
            rapidcsv::Document doc(path, label_params, sep); // Инициализация Document, открытие файла
            auto count = (float)(doc.GetRowCount() * sample);

            // Чтение колонок во временные переменные
            auto full_source = doc.GetColumn<std::string>(source_name);
            auto full_target = doc.GetColumn<std::string>(target_name);
            auto full_edge_attr = doc.GetColumn<double>(edge_attr_name);
            if (sample == 1.0) {
                return { full_source, full_target, full_edge_attr };
            }

            // Создание срезов
            std::vector<std::string> source(full_source.begin(), full_source.begin() + count);
            std::vector<std::string> target(full_target.begin(), full_target.begin() + count);
            std::vector<double> edge_attr(full_edge_attr.begin(), full_edge_attr.begin() + count);

            return { source, target, edge_attr };
        }
        catch (const std::ios_base::failure& e) {
            std::cerr << "I/O Error: " << e.what() << std::endl;
            throw;
        }
        catch (const std::exception& ex) {
            std::cerr << "Error: " << ex.what() << std::endl;
            throw;
        }     
    }

    std::tuple<
        std::vector<std::string>,
        std::vector<std::string>,
        std::vector<double>>
        _extract_column(
            const std::string& path,
            int source_name,
            int target_name,
            int edge,
            rapidcsv::LabelParams label_params,
            double sample,
            rapidcsv::SeparatorParams sep)
    {
        try {
            rapidcsv::Document doc(path, label_params, sep); // Инициализация Document, открытие файла
            auto count = (float)(doc.GetRowCount() * sample);

            // Чтение колонок во временные переменные
            auto full_source = doc.GetColumn<std::string>(source_name);
            auto full_target = doc.GetColumn<std::string>(target_name);
            std::vector<double> full_edge_attr(full_source.size(), 1.0);
            if (sample == 1.0) {
                return { full_source, full_target, full_edge_attr };
            }

            // Создание срезов
            std::vector<std::string> source(full_source.begin(), full_source.begin() + count);
            std::vector<std::string> target(full_target.begin(), full_target.begin() + count);
            std::vector<double> edge_attr(full_edge_attr.begin(), full_edge_attr.begin() + count);

            return { source, target, edge_attr };
        }
        catch (const std::ios_base::failure& e) {
            std::cerr << "I/O Error: " << e.what() << std::endl;
            throw;
        }
        catch (const std::exception& ex) {
            std::cerr << "Error: " << ex.what() << std::endl;
            throw;
        }
    }


    TransitionMatrixBuilder& TransitionMatrixBuilder::SetPath(const std::string& path) {
        path_ = path;
        return *this;
    }

    TransitionMatrixBuilder& TransitionMatrixBuilder::SetSourceName(const std::string& source_name) {
        source_name_ = source_name;
        return *this;
    }

    TransitionMatrixBuilder& TransitionMatrixBuilder::SetTargetName(const std::string& target_name) {
        target_name_ = target_name;
        return *this;
    }

    TransitionMatrixBuilder& TransitionMatrixBuilder::SetEdgeAttributeName(
        const std::string& edge_attr_name) {
        edge_attr_name_ = edge_attr_name;
        return *this;
    }

    TransitionMatrixBuilder& TransitionMatrixBuilder::SetDirectional(bool directional) {
        directional_ = directional;
        return *this;
    }

    TransitionMatrixBuilder& TransitionMatrixBuilder::SetLabelParams(
        const rapidcsv::LabelParams& label_params) {
        label_params_ = label_params;
        return *this;
    }

    TransitionMatrixBuilder& TransitionMatrixBuilder::SetSample(double sample) {
        this->sample = sample;
        return *this;
    }

    TransitionMatrixBuilder& TransitionMatrixBuilder::SetSeparator(rapidcsv::SeparatorParams sep)
    {
        this->sep = sep;
        return *this;
    }

    std::tuple<SparseMatrix<double>, std::vector<std::pair<std::string, Type>>>
        TransitionMatrixBuilder::Build() const {
        auto [source, target, edge_attr] = _extract_column(
            path_, source_name_,
            target_name_, edge_attr_name_,
            label_params_, sample, sep);


        auto [object_source_to_integer, object_target_to_integer, integer_to_object] =
            _get_mapping(source, target);

        auto x = _get_mapping(source, target);

        auto triplets = _get_triplets(object_source_to_integer,
            object_target_to_integer,
            source, target, edge_attr, false);

        SparseMatrix<double> A(integer_to_object.size(), integer_to_object.size());
        A.setFromTriplets(triplets.begin(), triplets.end());

        auto rowSums = _get_sum_matrix_rows(A);

        _norm_matrix_rows(A, rowSums);

        return { A, integer_to_object };
    }

    Eigen::VectorXd get_pagerank(const SparseMatrix<double>& trasition,
        const Eigen::VectorXd& person_vec,
        double a, double b, double c, int max_iter) 
    {
        assert(a + b + c == 1.0);
        auto transtion_T_mult_a = (trasition.transpose() * a).eval();
        auto person_vec_mult_b = (person_vec * b).eval();
        auto sup_vec = (Eigen::VectorXd::Ones(transtion_T_mult_a.rows()) /
            transtion_T_mult_a.rows() * c).eval();
        auto pagerank = (Eigen::VectorXd::Ones(transtion_T_mult_a.rows()) /
            transtion_T_mult_a.rows()).eval();
        for (int i = 0; i < max_iter; i++) {
            pagerank = transtion_T_mult_a * pagerank + person_vec_mult_b * pagerank + sup_vec;
        }
        return pagerank;
    }

    Eigen::VectorXd get_pagerank(const SparseMatrix<double>& trasition,
        const Eigen::VectorXd& person_vec,
        double a, double b, double c, int max_iter);

    PageRankFactory& PageRankFactory::SetTransitionMatrix(const SparseMatrix<double>& matrix) {
        transition_matrix_ = &matrix;
        return *this;
    }

    PageRankFactory& PageRankFactory::SetPersonalizationVector(const Eigen::VectorXd& vector) {
        personalization_vector_ = &vector;
        return *this;
    }

    PageRankFactory& PageRankFactory::SetAlpha(double alpha) {
        alpha_ = alpha;
        return *this;
    }

    PageRankFactory& PageRankFactory::SetBeta(double beta) {
        beta_ = beta;
        return *this;
    }

    PageRankFactory& PageRankFactory::SetGamma(double gamma) {
        gamma_ = gamma;
        return *this;
    }

    PageRankFactory& PageRankFactory::SetMaxIterations(int max_iterations) {
        max_iterations_ = max_iterations;
        return *this;
    }



    Eigen::VectorXd PageRankFactory::Build() const {
        assert(alpha_ + beta_ + gamma_ == 1.0);
        return get_pagerank(
            *transition_matrix_,
            *personalization_vector_,
            alpha_, beta_, gamma_,
            max_iterations_);
    }

    Eigen::VectorXd get_personalization_from_matrix_row(
        const SparseMatrix<double>& matrix,
        int index) 
    {
        // Создаем плотный вектор той же размерности, что и строка
        Eigen::VectorXd dense_vec(matrix.cols());

        // Копируем ненулевые элементы из строки в плотный вектор
        for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, index); it; ++it) {
            dense_vec[it.col()] = it.value();
        }
        return dense_vec;
    }

    Eigen::SparseMatrix<double> multiply_rows_by_scalar_from_vector(
        const Eigen::SparseMatrix<double>& matrix,
        const Eigen::VectorXd& scalars) {

        assert(matrix.rows() == scalars.size());

        // Создаем копию матрицы для результата
        Eigen::SparseMatrix<double> result = matrix;

        
        // Итерируемся по строкам разреженной матрицы
        for (int i = 0; i < result.rows(); ++i) {
            // Умножаем ненулевые элементы строки на соответствующий скаляр
            for (Eigen::SparseMatrix<double>::InnerIterator it(result, i); it; ++it) {
                it.valueRef() *= scalars(i);
            }
        }
        


        return result;
    }


    std::tuple<Eigen::VectorXd, Eigen::VectorXd>  
        get_hits(const SparseMatrix<double>& trasition, int max_iter)
    {

        auto transtion_T = trasition.transpose().eval();
        auto hub = (Eigen::VectorXd::Ones(transtion_T.rows()) /
            transtion_T.rows()).eval();
        auto auth = (Eigen::VectorXd::Ones(trasition.rows()) /
            trasition.rows()).eval();

        for (int i = 0; i < max_iter; i++) {
            auto old_hub = hub / hub.sum();
            auto old_auth = auth / auth.sum();
      
            hub = _get_sum_matrix_rows(
                multiply_rows_by_scalar_from_vector(trasition, old_auth));
            auth = _get_sum_matrix_rows(
                multiply_rows_by_scalar_from_vector(transtion_T, old_hub));
        }
        return { auth, hub };
    }


    HITSFactory& HITSFactory::SetTransitionMatrix(const Eigen::SparseMatrix<double>& matrix) {
        trasition = &matrix;
        return *this;
    }

    HITSFactory& HITSFactory::SetMaxIter(int iterations) {
        max_iter = iterations;
        return *this;
    }


    std::tuple<Eigen::VectorXd, Eigen::VectorXd> HITSFactory::Build() const {
        return get_hits(*trasition, max_iter);
    }
    
    TransitionMatrixFromTxtBuilder& TransitionMatrixFromTxtBuilder::SetPath(const std::string& path)
    {
        this->path_ = path;
        return *this;
    }

    TransitionMatrixFromTxtBuilder& TransitionMatrixFromTxtBuilder::SetSourceName(int source_name)
    {
        this->source = source_name;
        return *this;
    }

    TransitionMatrixFromTxtBuilder& TransitionMatrixFromTxtBuilder::SetTargetName(int target_name)
    {
        this->target = target_name;
        return *this;
    }

    TransitionMatrixFromTxtBuilder& TransitionMatrixFromTxtBuilder::SetEdgeAttributeName(int edge_attr_name)
    {
        this->edge = edge_attr_name;
        return *this;
    }

    TransitionMatrixFromTxtBuilder& TransitionMatrixFromTxtBuilder::SetDirectional(bool directional)
    {
        this->directional_ = directional;
        return *this;
    }

    TransitionMatrixFromTxtBuilder& TransitionMatrixFromTxtBuilder::SetLabelParams(const rapidcsv::LabelParams& label_params)
    {
        this->label_params_ = label_params;
        return *this;
    }

    TransitionMatrixFromTxtBuilder& TransitionMatrixFromTxtBuilder::SetSample(double sample)
    {
        this->sample = sample;
        return *this;
    }

    TransitionMatrixFromTxtBuilder& TransitionMatrixFromTxtBuilder::SetSeparator(rapidcsv::SeparatorParams sep)
    {
        this->sep = sep;
        return *this;
    }


    std::tuple<Eigen::SparseMatrix<double>, std::vector<std::pair<std::string, Type>>> TransitionMatrixFromTxtBuilder::Build() const
    {
        auto [source, target, edge_attr] = _extract_column(
            path_, this->source,
            this->target, this->edge,
            label_params_, sample, sep);


        auto [object_source_to_integer, object_target_to_integer, integer_to_object] =
            _get_mapping(source, target);

        auto x = _get_mapping(source, target);

        auto triplets = _get_triplets(object_source_to_integer,
            object_target_to_integer,
            source, target, edge_attr, false);

        SparseMatrix<double> A(integer_to_object.size(), integer_to_object.size());
        A.setFromTriplets(triplets.begin(), triplets.end());

        auto rowSums = _get_sum_matrix_rows(A);

        _norm_matrix_rows(A, rowSums);

        return { A, integer_to_object };
        return std::tuple<Eigen::SparseMatrix<double>, std::vector<std::pair<std::string, Type>>>();
    }

}

