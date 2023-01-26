//
// Created by ramizouari on 24/01/23.
//

#ifndef COUNTERSTRATEGY_TROPICAL_MATRIX_H
#define COUNTERSTRATEGY_TROPICAL_MATRIX_H

#include <vector>
#include "extended_integer.h"
#include "operation.h"

/**
 * @brief A class representing a tropical matrix
 * @details It is a matrix of extended integers, on which matrix multiplication is defined as:
 * (A*B)(i,j)=min_{k} (A(i,k)+B(k,j))
 * */
struct TropicalMatrix
{
    std::vector<std::vector<ExtendedInteger>> matrix;
    explicit TropicalMatrix(size_t n,size_t m): matrix(n,std::vector<ExtendedInteger>(m,inf_p))
    {}
    explicit TropicalMatrix(size_t n): TropicalMatrix(n,n)
    {}
    TropicalMatrix(std::vector<std::vector<ExtendedInteger>> &&matrix): matrix(std::move(matrix))
    {}
    TropicalMatrix(const TropicalMatrix& other): matrix(other.matrix)
    {}
    TropicalMatrix(): TropicalMatrix(0,0)
    {}

    bool operator==(const TropicalMatrix& other) const = default;

    ExtendedInteger& operator()(size_t i,size_t j)
    {
        return matrix[i][j];
    }

    const ExtendedInteger& operator()(size_t i,size_t j) const
    {
        return matrix[i][j];
    }

    TropicalMatrix& operator+=(const TropicalMatrix& other)
    {
        for(size_t i=0;i<std::min(matrix.size(),other.matrix.size());i++)
            for(size_t j=0;j<std::min(matrix[i].size(),other.matrix[i].size());j++)
                matrix[i][j] += other.matrix[i][j];
        return *this;
    }

    TropicalMatrix operator+(const TropicalMatrix& other) const
    {
        TropicalMatrix result(*this);
        result += other;
        return result;
    }

    TropicalMatrix operator*(const TropicalMatrix& other) const
    {
        if(other.matrix.empty() || other.matrix[0].empty())
            return *this;
        auto n = matrix.size();
        auto m = other.matrix[0].size();
        auto r = matrix[0].size();
        TropicalMatrix result(n,m);
        for(int i=0;i<n;i++) for(int k=0;k<r;k++) for(int j=0;j<m;j++)
                    result(i,j) = std::min(result(i,j),matrix[i][k]+other.matrix[k][j]);
        return result;
    }

    virtual std::vector<ExtendedInteger> operator*(const std::vector<ExtendedInteger>& other) const
    {
        auto n = matrix.size();
        auto m = other.size();
        std::vector<ExtendedInteger> result(n,inf_p);
        for(int i=0;i<n;i++) for(int j=0;j<m;j++)
                result[i] = std::min(result[i],matrix[i][j]+other[j]);
        return result;
    }

};

/**
 * @brief A class representing a tropical matrix with a mask
 * @details It is a matrix of extended integers, on which matrix multiplication is defined as:
 * (A*B)(i,j)=min_{k / A.mask(i,k) ∧ B.mask(k,j) } (A(i,k)+B(k,j))
 * */
struct MaskedTropicalMatrix : public TropicalMatrix
{
    std::vector<std::vector<bool>> mask;
    explicit MaskedTropicalMatrix(size_t n,size_t m,std::vector<std::vector<bool>> mask): TropicalMatrix(n,m), mask(std::move(mask))
    {}
    explicit MaskedTropicalMatrix(size_t n,size_t m): MaskedTropicalMatrix(n,m,std::vector<std::vector<bool>>(n,std::vector<bool>(m,true)))
    {}
    explicit MaskedTropicalMatrix(size_t n): MaskedTropicalMatrix(n,n)
    {}
    MaskedTropicalMatrix(std::vector<std::vector<ExtendedInteger>> &&matrix,std::vector<std::vector<bool>> &&mask): TropicalMatrix(std::move(matrix)), mask(std::move(mask))
    {}
    MaskedTropicalMatrix(): MaskedTropicalMatrix(0,0)
    {}

    MaskedTropicalMatrix operator*(const MaskedTropicalMatrix& other) const
    {
        if(other.matrix.empty() || other.matrix[0].empty())
            return *this;
        auto n = matrix.size();
        auto m = other.matrix[0].size();
        auto r = matrix[0].size();
        std::vector<std::vector<bool>> resultMask(n,std::vector<bool>(m,false));
        for(int i=0;i<n;i++) for(int k=0;k<r;k++) for(int j=0;j<m;j++)
                    resultMask[i][j] = resultMask[i][j] || (mask[i][k] && other.mask[k][j]);
        MaskedTropicalMatrix result(n,m,resultMask);
        for(int i=0;i<n;i++) for(int k=0;k<r;k++) for(int j=0;j<m;j++) if(mask[i][k] && other.mask[k][j])
                    result(i,j) = std::min(result(i,j),matrix[i][k]+other.matrix[k][j]);
        return result;
    }

    std::vector<ExtendedInteger> operator*(const std::vector<ExtendedInteger>& other) const override
    {
        auto n = matrix.size();
        auto m = other.size();
        std::vector<ExtendedInteger> result(n,inf_plus_t{});
        for(int i=0;i<n;i++) for(int j=0;j<m;j++) if(mask[i][j])
                result[i] = std::min(result[i],matrix[i][j]+other[j]);
        return result;
    }
};


template<std::derived_from<TropicalMatrix> TropicalMatrix_t>
struct Multiplication<TropicalMatrix_t> : public BinaryOperation<TropicalMatrix_t>
{
    [[nodiscard]] TropicalMatrix_t reduce(const TropicalMatrix_t& u,const TropicalMatrix_t& v) const override
    {
        return u*v;
    }
    /**
     * @brief The identity element of the tropical matrix multiplication
     * @details It is a matrix of zeros.
     * For convenience, it is declared as a matrix with size 0x0
     * */
    inline static TropicalMatrix_t identity{0,0};
};

#endif //COUNTERSTRATEGY_TROPICAL_MATRIX_H
