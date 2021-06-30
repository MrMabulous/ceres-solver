// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/triplet_sparse_matrix.h"

#include <algorithm>
#include <cstddef>

#include "ceres/internal/eigen.h"
#include "ceres/internal/port.h"
#include "ceres/random.h"
#include "ceres/types.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

TripletSparseMatrix::TripletSparseMatrix(bool force_long_indices)
    : num_rows_(0),
    num_cols_(0),
    max_num_nonzeros_(0),
    num_nonzeros_(0),
    index_type_(force_long_indices ? INT32 : INT64) {}

TripletSparseMatrix::~TripletSparseMatrix() {}

TripletSparseMatrix::TripletSparseMatrix(int64_t num_rows,
                                         int64_t num_cols,
                                         int64_t max_num_nonzeros,
                                         bool force_long_indices)
    : num_rows_(num_rows),
      num_cols_(num_cols),
      max_num_nonzeros_(max_num_nonzeros),
      num_nonzeros_(0) {
  // All the sizes should at least be zero
  CHECK_GE(num_rows, 0);
  CHECK_GE(num_cols, 0);
  CHECK_GE(max_num_nonzeros, 0);
  index_type_ = force_long_indices
      || max_num_nonzeros_ >= std::numeric_limits<int>::max()) ? INT64
                                                               : INT32;
  AllocateMemory();
}

template<typename T>
TripletSparseMatrix::TripletSparseMatrix(const int64_t num_rows,
                                         const int64_t num_cols,
                                         const std::vector<T>& rows,
                                         const std::vector<T>& cols,
                                         const std::vector<double>& values,
                                         bool force_long_indices)
    : num_rows_(num_rows),
      num_cols_(num_cols),
      max_num_nonzeros_(values.size()),
      num_nonzeros_(values.size()) {
  // All the sizes should at least be zero
  CHECK_GE(num_rows, 0);
  CHECK_GE(num_cols, 0);
  CHECK_EQ(rows.size(), cols.size());
  CHECK_EQ(rows.size(), values.size());
  index_type_ = force_long_indices
      || max_num_nonzeros_ >= std::numeric_limits<int>::max()) ? INT64
                                                               : INT32;
  AllocateMemory();
  if(index_type_ == INT32) {
    for(int i = 0; i < values.size(); ++i) {
      rows_[i] = static_cast<int>(rows[i]);
      cols_[i] = static_cast<int>(cols[i]);
    }
  } else {
    for(int64_t i = 0; i < values.size(); ++i) {
      rows64_[i] = static_cast<int64_t>(rows[i]);
      cols64_[i] = static_cast<int64_t>(cols[i]);
    }
  }
  std::copy(values.begin(), values.end(), values_.get());
}

TripletSparseMatrix::TripletSparseMatrix(
    const TripletSparseMatrix& orig)
    : SparseMatrix(),
      num_rows_(orig.num_rows_),
      num_cols_(orig.num_cols_),
      max_num_nonzeros_(orig.max_num_nonzeros_),
      num_nonzeros_(orig.num_nonzeros_),
      index_type_(orig.index_type_) {
  AllocateMemory();
  CopyData(orig);
}

TripletSparseMatrix& TripletSparseMatrix::operator=(
    const TripletSparseMatrix& rhs) {
  if (this == &rhs) {
    return *this;
  }
  num_rows_ = rhs.num_rows_;
  num_cols_ = rhs.num_cols_;
  num_nonzeros_ = rhs.num_nonzeros_;
  max_num_nonzeros_ = rhs.max_num_nonzeros_;
  index_type_ = rhs.index_type_;
  AllocateMemory();
  CopyData(rhs);
  return *this;
}

bool TripletSparseMatrix::AllTripletsWithinBounds() const {
  if(index_type_ == INT32) {
    for (int i = 0; i < num_nonzeros_; ++i) {
      // clang-format off
      if ((rows_[i] < 0) || (rows_[i] >= num_rows_) ||
          (cols_[i] < 0) || (cols_[i] >= num_cols_))
        return false;
      // clang-format on
    }
  } else {
    for (int64_t i = 0; i < num_nonzeros_; ++i) {
      // clang-format off
      if ((rows64_[i] < 0) || (rows64_[i] >= num_rows_) ||
          (cols64_[i] < 0) || (cols64_[i] >= num_cols_))
        return false;
      // clang-format on
    }
  }
  return true;
}

void TripletSparseMatrix::Reserve(int64_t new_max_num_nonzeros) {
  CHECK_LE(num_nonzeros_, new_max_num_nonzeros)
      << "Reallocation will cause data loss";

  // Nothing to do if we have enough space already.
  if (new_max_num_nonzeros <= max_num_nonzeros_) return;

  IndexType new_index_type = index_type_ == INT64 ||
      new_max_num_nonzeros >= std::numeric_limits<int>::max()) ? INT64
                                                               : INT32;
  double* new_values = new double[new_max_num_nonzeros];
  if(new_index_type == INT32) {
    CHECK_EQ(index_type_, INT32);
    int* new_rows = new int[new_max_num_nonzeros];
    int* new_cols = new int[new_max_num_nonzeros];

    for (int64_t i = 0; i < num_nonzeros_; ++i) {
      new_rows[i] = rows_[i];
      new_cols[i] = cols_[i];
      new_values[i] = values_[i];
    }

    rows_.reset(new_rows);
    cols_.reset(new_cols);
    rows64_.reset(nullptr);
    cols64_.reset(nullptr);
  } else {
    int64_t* new_rows = new int64_t[new_max_num_nonzeros];
    int64_t* new_cols = new int64_t[new_max_num_nonzeros];
    
    if(index_type_ == INT32) {
      for (int64_t i = 0; i < num_nonzeros_; ++i) {
        new_rows[i] = static_cast<int64_t>(rows_[i]);
        new_cols[i] = static_cast<int64_t>(cols_[i]);
        new_values[i] = values_[i];
      }
    } else {
      for (int64_t i = 0; i < num_nonzeros_; ++i) {
        new_rows[i] = rows64_[i];
        new_cols[i] = cols64_[i];
        new_values[i] = values_[i];
      }
    }

    rows_.reset(nullptr);
    cols_.reset(nullptr);
    rows64_.reset(new_rows);
    cols64_.reset(new_cols);
  }

  values_.reset(new_values);
  max_num_nonzeros_ = new_max_num_nonzeros;
  index_type_ = new_index_type;
}

void TripletSparseMatrix::SetZero() {
  std::fill(values_.get(), values_.get() + max_num_nonzeros_, 0.0);
  num_nonzeros_ = 0;
}

void TripletSparseMatrix::set_num_nonzeros(int64_t num_nonzeros) {
  CHECK_GE(num_nonzeros, 0);
  CHECK_LE(num_nonzeros, max_num_nonzeros_);
  num_nonzeros_ = num_nonzeros;
}

void TripletSparseMatrix::AllocateMemory() {
  if(index_type_ == INT32) {
    rows_.reset(new int[max_num_nonzeros_]);
    cols_.reset(new int[max_num_nonzeros_]);
  } else {
    rows64_.reset(new int64_t[max_num_nonzeros_]);
    cols64_.reset(new int64_t[max_num_nonzeros_]);
  }
  values_.reset(new double[max_num_nonzeros_]);
}

void TripletSparseMatrix::CopyData(
    const TripletSparseMatrix& orig) {
  CHECK_EQ(index_type_, orig.index_type_);
  if(index_type_ == INT32) {
    for (int64_t i = 0; i < num_nonzeros_; ++i) {
      rows_[i] = orig.rows_[i]);
      cols_[i] = orig.cols_[i]);
      values_[i] = orig.values_[i];
    }
  } else {
    for (int64_t i = 0; i < num_nonzeros_; ++i) {
      rows64_[i] = orig.rows64_[i]);
      cols64_[i] = orig.cols64_[i]);
      values_[i] = orig.values_[i];
    }
  }
}

void TripletSparseMatrix::RightMultiply(const double* x, double* y) const {
  if(index_type_ == INT32) {
    for (int64_t i = 0; i < num_nonzeros_; ++i) {
      y[rows_[i]] += values_[i] * x[cols_[i]];
    }
  } else {
    for (int64_t i = 0; i < num_nonzeros_; ++i) {
      y[rows64_[i]] += values_[i] * x[cols64_[i]];
    }
  }
}

void TripletSparseMatrix::LeftMultiply(const double* x, double* y) const {
  if(index_type_ == INT32) {
    for (int64_t i = 0; i < num_nonzeros_; ++i) {
      y[cols_[i]] += values_[i] * x[rows_[i]];
    }
  } else {
    for (int64_t i = 0; i < num_nonzeros_; ++i) {
      y[cols64_[i]] += values_[i] * x[rows64_[i]];
    }
  }
}

void TripletSparseMatrix::SquaredColumnNorm(double* x) const {
  CHECK(x != nullptr);
  memset(x, 0, sizeof(double) * num_cols_);
  if(index_type_ == INT32) {
    for (int64_t i = 0; i < num_nonzeros_; ++i) {
      x[cols_[i]] += values_[i] * values_[i];
    }
  } else {
    for (int64_t i = 0; i < num_nonzeros_; ++i) {
      x[cols64_[i]] += values_[i] * values_[i];
    }
  }
}

void TripletSparseMatrix::ScaleColumns(const double* scale) {
  CHECK(scale != nullptr);
  if(index_type_ == INT32) {
    for (int64_t i = 0; i < num_nonzeros_; ++i) {
      values_[i] = values_[i] * scale[cols_[i]];
    }
  } else {
    for (int64_t i = 0; i < num_nonzeros_; ++i) {
      values_[i] = values_[i] * scale[cols64_[i]];
    }
  }
}

void TripletSparseMatrix::ToDenseMatrix(Matrix* dense_matrix) const {
  CHECK_LE(num_rows_, std::numeric_limits<Eigen::Index>::max());
  CHECK_LE(num_cols_, std::numeric_limits<Eigen::Index>::max());
  dense_matrix->resize(static_cast<Eigen::Index>(num_rows_),
                       static_cast<Eigen::Index>(num_cols_));
  dense_matrix->setZero();
  Matrix& m = *dense_matrix;
  if(index_type_ == INT32) {
    for (int64_t i = 0; i < num_nonzeros_; ++i) {
      m(rows_[i], cols_[i]) += values_[i];
    }
  } else {
    for (int64_t i = 0; i < num_nonzeros_; ++i) {
      m(rows64_[i], cols64_[i]) += values_[i];
    }
  }
}

void TripletSparseMatrix::AppendRows(const TripletSparseMatrix& B) {
  CHECK_EQ(B.num_cols(), num_cols_);
  Reserve(num_nonzeros_ + B.num_nonzeros_);
  if(index_type_ == INT32) {
    if(B.index_type() == INT32) {
      for (int64_t i = 0; i < B.num_nonzeros_; ++i) {
        rows_[num_nonzeros_] = reinterpret_cast<int*>(B.rows())[i] + num_rows_;
        cols_[num_nonzeros_] = reinterpret_cast<int*>(B.cols())[i];
        values_[num_nonzeros_++] = B.values()[i];
      }
    } else {
      for (int64_t i = 0; i < B.num_nonzeros_; ++i) {
        rows_[num_nonzeros_] = static_cast<int>(reinterpret_cast<int64_t*>(
            B.rows())[i] + num_rows_);
        cols_[num_nonzeros_] = static_cast<int>(reinterpret_cast<int64_t*>(
            B.cols())[i]);
        values_[num_nonzeros_++] = B.values()[i];
      }
    }
  } else {
    if(B.index_type() == INT32) {
      for (int64_t i = 0; i < B.num_nonzeros_; ++i) {
        rows64_[num_nonzeros_] = static_cast<int64_t>(reinterpret_cast<int*>(
            B.rows())[i] + num_rows_);
        cols64_[num_nonzeros_] = static_cast<int64_t>(reinterpret_cast<int*>(
            B.cols())[i]);
        values_[num_nonzeros_++] = B.values()[i];
      }
    } else {
      for (int64_t i = 0; i < B.num_nonzeros_; ++i) {
        rows64_[num_nonzeros_] =
            reinterpret_cast<int64_t*>(B.rows())[i] + num_rows_;
        cols64_[num_nonzeros_] = reinterpret_cast<int64_t*>(B.cols())[i];
        values_[num_nonzeros_++] = B.values()[i];
      }
    }
  }
  num_rows_ = num_rows_ + B.num_rows();
}

void TripletSparseMatrix::AppendCols(const TripletSparseMatrix& B) {
  CHECK_EQ(B.num_rows(), num_rows_);
  Reserve(num_nonzeros_ + B.num_nonzeros_);
  if(index_type_ == INT32) {
    if(B.index_type() == INT32) {
      for (int64_t i = 0; i < B.num_nonzeros_; ++i, ++num_nonzeros_) {
        rows_[num_nonzeros_] = reinterpret_cast<int*>(B.rows())[i];
        cols_[num_nonzeros_] = reinterpret_cast<int*>(B.cols())[i] + num_cols_;
        values_[num_nonzeros_] = B.values()[i];
      }
    } else {
      for (int64_t i = 0; i < B.num_nonzeros_; ++i, ++num_nonzeros_) {
        rows_[num_nonzeros_] = static_cast<int>(reinterpret_cast<int64_t*>(
            B.rows())[i]);
        cols_[num_nonzeros_] = static_cast<int>(reinterpret_cast<int64_t*>(
            B.cols())[i] + num_cols_);
        values_[num_nonzeros_] = B.values()[i];
      }
    }
  } else {
    if(B.index_type() == INT32) {
      for (int64_t i = 0; i < B.num_nonzeros_; ++i, ++num_nonzeros_) {
        rows64_[num_nonzeros_] = static_cast<int64_t>(reinterpret_cast<int*>(
            B.rows())[i]);
        cols64_[num_nonzeros_] = static_cast<int64_t>(reinterpret_cast<int*>(
            B.cols())[i] + num_cols_);
        values_[num_nonzeros_] = B.values()[i];
      }
    } else {
      for (int64_t i = 0; i < B.num_nonzeros_; ++i, ++num_nonzeros_) {
        rows64_[num_nonzeros_] = reinterpret_cast<int64_t*>(B.rows())[i];
        cols64_[num_nonzeros_] = reinterpret_cast<int64_t*>(
            B.cols())[i] + num_cols_;
        values_[num_nonzeros_] = B.values()[i];
      }
    }
  }
  num_cols_ = num_cols_ + B.num_cols();
}

void TripletSparseMatrix::Resize(int64_t new_num_rows,
                                 int64_t new_num_cols) {
  if ((new_num_rows >= num_rows_) && (new_num_cols >= num_cols_)) {
    num_rows_ = new_num_rows;
    num_cols_ = new_num_cols;
    return;
  }

  num_rows_ = new_num_rows;
  num_cols_ = new_num_cols;

  int* r_ptr = rows_.get();
  int* c_ptr = cols_.get();
  int64_t* r64_ptr = rows64_.get();
  int64_t* c64_ptr = cols64_.get();
  double* v_ptr = values_.get();

  int dropped_terms = 0;
  for (int64_t i = 0; i < num_nonzeros_; ++i) {
    if ((r_ptr[i] < num_rows_) && (c_ptr[i] < num_cols_)) {
      if (dropped_terms) {
        if(index_type_ == INT32) {
          r_ptr[i - dropped_terms] = r_ptr[i];
          c_ptr[i - dropped_terms] = c_ptr[i];
        } else {
          r64_ptr[i - dropped_terms] = r64_ptr[i];
          c64_ptr[i - dropped_terms] = c64_ptr[i];
        }
        v_ptr[i - dropped_terms] = v_ptr[i];
      }
    } else {
      ++dropped_terms;
    }
  }
  num_nonzeros_ -= dropped_terms;
  //TODO(matthias.buehlmann): Decide whether to downconvert to int32 indices.
}

const void* TripletSparseMatrix::rows() const {
  if(index_type_ == INT32) {
    return rows_.get();
  } else {
    return rows64_.get();
  }
}

const void* TripletSparseMatrix::cols() const {
  if(index_type_ == INT32) {
    return cols_.get();
  } else {
    return cols64_.get();
  }
}

void* TripletSparseMatrix::mutable_rows() {
  if(index_type_ == INT32) {
    return rows_.get();
  } else {
    return rows64_.get();
  }
}

void* TripletSparseMatrix::mutable_cols() {
  if(index_type_ == INT32) {
    return cols_.get();
  } else {
    return cols64_.get();
  }
}

TripletSparseMatrix* TripletSparseMatrix::CreateSparseDiagonalMatrix(
    const double* values, int64_t num_rows) {
  TripletSparseMatrix* m =
      new TripletSparseMatrix(num_rows, num_rows, num_rows);
  for (int64_t i = 0; i < num_rows; ++i) {
    if(m->index_type() == INT32) {
      reinterpret_cast<int*>(m->mutable_rows())[i] = static_cast<int>(i);
      reinterpret_cast<int*>(m->mutable_cols())[i] = static_cast<int>(i);
    } else {
      reinterpret_cast<int64_t*>(m->mutable_rows())[i] = i;
      reinterpret_cast<int64_t*>(m->mutable_cols())[i] = i;
    }
    m->mutable_values()[i] = values[i];
  }
  m->set_num_nonzeros(num_rows);
  return m;
}

void TripletSparseMatrix::ToTextFile(FILE* file) const {
  CHECK(file != nullptr);
  for (int64_t i = 0; i < num_nonzeros_; ++i) {
    if(index_type_ == INT32) {
      fprintf(file, "% 10ld % 10ld %17f\n", rows_[i], cols_[i], values_[i]);
    } else {
      fprintf(file, "% 10lld % 10lld %17f\n",
              static_cast<long long>(rows64_[i]),
              static_cast<long long>(cols64_[i]), values_[i]);      
    }
  }
}

TripletSparseMatrix* TripletSparseMatrix::CreateRandomMatrix(
    const TripletSparseMatrix::RandomMatrixOptions& options) {
  CHECK_GT(options.num_rows, 0);
  CHECK_GT(options.num_cols, 0);
  CHECK_GT(options.density, 0.0);
  CHECK_LE(options.density, 1.0);

  std::vector<int64_t> rows;
  std::vector<int64_t> cols;
  std::vector<double> values;
  while (rows.empty()) {
    rows.clear();
    cols.clear();
    values.clear();
    for (int64_t r = 0; r < options.num_rows; ++r) {
      for (int64_t c = 0; c < options.num_cols; ++c) {
        if (RandDouble() <= options.density) {
          rows.push_back(r);
          cols.push_back(c);
          values.push_back(RandNormal());
        }
      }
    }
  }

  return new TripletSparseMatrix(
      options.num_rows, options.num_cols, rows, cols, values);
}

}  // namespace internal
}  // namespace ceres
