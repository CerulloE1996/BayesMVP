
#pragma once

 
  

#include <tbb/concurrent_vector.h>
 
#include <RcppParallel.h>
 
#include <Eigen/Dense>
 
 
 
 
#if defined(__AVX2__) || defined(__AVX512F__) 
#include <immintrin.h>
#endif
 
 

 
 
using namespace Eigen;
using namespace Rcpp;






 





inline Eigen::Matrix<double, -1, 1> fn_convert_RMatrixColumn_to_EigenColVec(const RcppParallel::RMatrix<double>::Column &rcpp_col) {
  
  Eigen::Matrix<double, -1, 1> eigen_col_vec(rcpp_col.size());
  
   
  for (int i = 0; i < rcpp_col.size(); ++i) {
    eigen_col_vec(i) = rcpp_col[i];
  } 
  
  return eigen_col_vec;
} 





inline Eigen::Matrix<double, -1, 1> fn_convert_RCppNumMat_Column_to_EigenColVec(const Rcpp::NumericMatrix::Column &rcpp_col) {
  
  // Rcpp::NumericVector col_vec = rcpp_col;
  
  Eigen::Matrix<double, -1, 1> eigen_col_vec(rcpp_col.size());
  
  for (int i = 0; i < rcpp_col.size(); ++i) {
    eigen_col_vec(i) = rcpp_col[i];
  }  
  
  // Eigen::Matrix<double, -1, 1> eigen_col_vec = ;
  return eigen_col_vec;
  
} 





// Function to convert Eigen column vector to RcppParallel::RMatrix column
inline RcppParallel::RMatrix<double>::Column   fn_convert_EigenColVec_to_RMatrixColumn(const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> eigen_col_vec,  // const 
                                                                                       RcppParallel::RMatrix<double>::Column   r_matrix_col // not const
) { 
  
  // Ensure the size of the Eigen column vector matches the size of the RMatrix column
  if (eigen_col_vec.rows() != r_matrix_col.size()) {
    throw std::invalid_argument("Eigen column vector and RMatrix column must have the same number of rows.");
  }   
  
  // Copy data from Eigen column vector to RMatrix column
  for (int i = 0; i < eigen_col_vec.rows(); ++i) {
    r_matrix_col[i] = eigen_col_vec(i); 
  }  
  
  
  return r_matrix_col;
  
}  





inline void copy_to_global(int chain_idx, 
                            int n_iter, 
                            const Eigen::Matrix<double, -1, -1> &local_buffer,
                            std::vector<Eigen::Matrix<double, -1, -1>> &global_trace) {
  
  // if (global_trace.size() <= chain_idx) {
  //   global_trace.resize(chain_idx + 1);
  // }
  
  global_trace[chain_idx] = local_buffer;
  
}





inline void copy_to_global_float(int chain_idx, 
                                 int n_iter, 
                                 const Eigen::Matrix<float, -1, -1> &local_buffer,
                                 std::vector<Eigen::Matrix<float, -1, -1>> &global_trace) {
  
  // if (global_trace.size() <= chain_idx) {
  //   global_trace.resize(chain_idx + 1);
  // }
  
  global_trace[chain_idx] = local_buffer;
  
}




// inline void copy_to_global_tbb(   int chain_index,
//                                   int n_iter,
//                                   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Eigen_thread_local_trace_buffer,
//                                   tbb::concurrent_vector<RcppParallel::RMatrix<double>>  &trace_output_to_R_RcppPar) {
//   
//   for (int ii = 0; ii < n_iter; ++ii) {  
//     for (int param = 0; param < Eigen_thread_local_trace_buffer.rows(); ++param) {
//       trace_output_to_R_RcppPar[chain_index](param, ii) = Eigen_thread_local_trace_buffer(param, ii);
//     }
//   } 
//   
// }  
// 


inline void copy_to_global_tbb(int chain_index,
                               int n_iter,
                               const Eigen::Matrix<double, -1, -1> &Eigen_thread_local_trace_buffer,
                               tbb::concurrent_vector<RcppParallel::RMatrix<double>> &trace_output_to_R_RcppPar) {
  
  auto &target_matrix = trace_output_to_R_RcppPar[chain_index];
  const int n_params = Eigen_thread_local_trace_buffer.rows();
  
  // Copy by columns, getting the actual data pointer
  for (int ii = 0; ii < n_iter; ++ii) {
    auto col = Eigen_thread_local_trace_buffer.col(ii);
    std::memcpy(&target_matrix(0, ii),
                col.data(),
                n_params * sizeof(double));
  }
}



 

 inline tbb::concurrent_vector<RcppParallel::RMatrix<double>>    convert_vec_of_RcppMat_to_concurrent_vector(const std::vector<Rcpp::NumericMatrix> &input_matrices,
                                                                                                            tbb::concurrent_vector<RcppParallel::RMatrix<double>> &result) {
   
  //  result.reserve(input_matrices.size());
  
  for (const auto &matrix : input_matrices) {
    result.emplace_back(RcppParallel::RMatrix<double>( 
        const_cast<double*>(&matrix[0]),  // Pointer to data
        matrix.nrow(),                    // Number of rows
        matrix.ncol()                     // Number of columns 
    ));
  }
  
  return result;
  
} 

 
 
template <typename T> 
inline tbb::concurrent_vector<T>    convert_std_vec_to_concurrent_vector(const std::vector<T> &inputs, 
                                                                                tbb::concurrent_vector<T> &result) {
   
   const int dim = inputs.size();
  
   /// tbb::concurrent_vector<T> result; 
  
   for (int i = 0; i < dim; ++i) {
     result.emplace_back(inputs[i]);
   }
   
   return result;
   
} 

 
 

 





 
 
 
inline void assign_column_to_3D_RcppNumericMat(   Rcpp::NumericMatrix &array_2D,
                                                  const Rcpp::NumericMatrix::Column &col_data,
                                                  const int dim1, const int dim2, const int dim3,
                                                  const int col_index) {
       
       if (col_index >= dim3) {
         Rcpp::stop("Invalid column index for 3D array.");
       }
       
       for (int j = 0; j < dim2; ++j) {
         for (int i = 0; i < dim1; ++i) {
           int row_index = flatten_index(i, j, dim1);
           array_2D(row_index, col_index) = col_data[row_index];
         }
       }
       
}
 
 
 
 
 
 
 


inline void assign_column_to_3D_RcppParRMatrix(  RcppParallel::RMatrix<double> &array_2D,
                                                const RcppParallel::RMatrix<double>::Column &col_data,
                                                const int dim1, const int dim2, const int dim3,
                                                const int col_index) {
     
     if (col_index >= dim3) {
       Rcpp::stop("Invalid column index for 3D array.");
     }
     
     // Loop over the first two dimensions and assign values from col_data
     for (int j = 0; j < dim2; ++j) {
       for (int i = 0; i < dim1; ++i) {
         int row_index = flatten_index(i, j, dim1);  // Flatten the indices
         array_2D(row_index, col_index) = col_data[row_index];  // Assign the value
       }
     }
     
}


 
 
 
 
 
 
 
 
inline  std::vector<Model_fn_args_struct> replicate_Model_fn_args_struct( const Model_fn_args_struct &input_struct,
                                                                                 int N) {
   
   std::vector<Model_fn_args_struct> replicated_structs;
   replicated_structs.reserve(N);  // Reserve space for N copies
   
   for (int i = 0; i < N; i++) { 
     replicated_structs.emplace_back(input_struct);  // Copy the input struct into each slot
   }  
   
   return replicated_structs;
   
}
 
 
 
 
 
inline std::vector<EHMC_fn_args_struct> replicate_EHMC_fn_args_struct( const EHMC_fn_args_struct &input_struct, /// these are NOT const 
                                                                              int N) {
   
   std::vector<EHMC_fn_args_struct> replicated_structs; 
   replicated_structs.reserve(N);  // Reserve space for N copies 
   
   for (int i = 0; i < N; i++) {
     replicated_structs.emplace_back(input_struct);  // Copy the input struct into each slot
   } 
   
   return replicated_structs;
   
}  
 
 
 
 
 
 
 
 
 
inline  std::vector<EHMC_Metric_struct> replicate_EHMC_Metric_struct( const EHMC_Metric_struct &input_struct,
                                                                             int N) { 
   
   std::vector<EHMC_Metric_struct> replicated_structs; 
   replicated_structs.reserve(N);  // Reserve space for N copies 
   
   for (int i = 0; i < N; i++) {
     replicated_structs.emplace_back(input_struct);  // Copy the input struct into each slot 
   }
   
   return replicated_structs;
   
}  
 
 
 
 
inline  std::vector<EHMC_burnin_struct> replicate_EHMC_burnin_struct( const EHMC_burnin_struct &input_struct,
                                                                              int N) {
   
   std::vector<EHMC_burnin_struct> replicated_structs;
   replicated_structs.reserve(N);  // Reserve space for N copies
   
   for (int i = 0; i < N; i++) { 
     replicated_structs.emplace_back(input_struct);  // Copy the input struct into each slot
   }   
   
   return replicated_structs;
   
}
 
 
 
 
 
 
 
 
 
 
 
// Function to convert an Rcpp::NumericMatrix to RcppParallel::RMatrix
inline RcppParallel::RMatrix<double> fn_convert_RcppMat_to_RMatrix_double(const Rcpp::NumericMatrix &input_matrix) {
   
  RcppParallel::RMatrix<double> rmatrix(input_matrix);
  
  return rmatrix;
  
} 



 



// // 
// Convert Eigen::Matrix<T, -1, 1> to RcppParallel::RVector
template <typename T>
inline RcppParallel::RVector<T> fn_convert_EigenColVec_to_RVec(Eigen::Ref<Eigen::Matrix<T, -1, 1>> EigenColVec) {
  
      RcppParallel::RVector<T> RVec(EigenColVec.rows());
      
      for (int i = 0; i < EigenColVec.rows(); ++i) {
        RVec[i] = EigenColVec(i);
      }
      
      return RVec;
  
}




// Convert RcppParallel::RVector to Eigen::Matrix<T, -1, 1>
template <typename T>
inline Eigen::Matrix<T, -1, 1> fn_convert_RVec_to_EigenColVec(RcppParallel::RVector<T> &RVec) {
  
      Eigen::Matrix<T, -1, 1> EigenColVec(RVec.size());
      
      for (int i = 0; i < RVec.size(); ++i) {
        EigenColVec(i) = RVec[i];
      }
      
      return EigenColVec;
  
}








// / ----------------
inline Eigen::Matrix<double, -1, 1>  fn_convert_RVec_to_EigenColVec_double(RcppParallel::RVector<double> &RVec) {
  
  Eigen::Matrix<double, -1, 1> EigenColVec(RVec.length());
  
  for (int i = 0; i < RVec.length(); ++i) {
    EigenColVec(i) = RVec[i];
  }
  
  return EigenColVec;
}





inline Eigen::Matrix<double, -1, 1> fn_convert_RcppNumericMatrixColumn_to_EigenColVec(Rcpp::NumericMatrix::Column &rcpp_col) {
  
  Eigen::Matrix<double, -1, 1> eigen_col_vec(rcpp_col.size());
  
  
  for (int i = 0; i < rcpp_col.size(); ++i) {
    eigen_col_vec(i) = rcpp_col[i]; 
  }
  
  return eigen_col_vec;
} 







// Function to convert Eigen column vector to RcppParallel::RMatrix column
inline Rcpp::NumericMatrix::Column                  fn_convert_EigenColVec_to_RcppNumericMatrixColumn(const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>  eigen_col_vec, 
                                                                                                      Rcpp::NumericMatrix::Column    &r_matrix_col) {
  
  // Ensure the size of the Eigen column vector matches the size of the RMatrix column
  if (eigen_col_vec.rows() != r_matrix_col.size()) {
    throw std::invalid_argument("Eigen column vector and Rcpp::NumericMatrix column must have the same number of rows.");
  }
  
  // Copy data from Eigen column vector to RMatrix column
  for (int i = 0; i < eigen_col_vec.rows(); ++i) {
    r_matrix_col[i] = eigen_col_vec(i);
  }
  
  return r_matrix_col;
  
}


 


// Function to convert Eigen::Matrix<double, -1, -1> to RcppParallel::RMatrix column by column
inline RcppParallel::RMatrix<double> fn_convert_EigenMat_to_RMatrix(const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> EigenMat,
                                                                    RcppParallel::RMatrix<double> &RMat) {
  int n_cols = EigenMat.cols();
  int n_rows = EigenMat.rows();
  
  for (int i = 0; i < n_cols; ++i) {
    
    RcppParallel::RMatrix<double>::Column RMatCol = RMat.column(i);
    
    for (int row = 0; row < n_rows; ++row) {
      RMatCol[row] = EigenMat(row, i);  // Assign values column by column
    }
    
    RMat.column(i) = RMatCol;
    
  }
  
  return RMat;
  
  
}


// Convert RcppParallel::RMatrix to Eigen::Matrix<double, -1, -1>
inline Eigen::Matrix<double, -1, -1> fn_convert_RMatrix_to_Eigen(const RcppParallel::RMatrix<double> &RMat) {
  
  Eigen::Matrix<double, -1, -1> EigenMat(RMat.nrow(), RMat.ncol());
  
  for (int j = 0; j < RMat.ncol(); ++j) { // col-major storage
   for (int i = 0; i < RMat.nrow(); ++i) {
      EigenMat(i, j) = RMat(i, j);
    }
  }
  
  return EigenMat; 
} 


// Convert RcppParallel::RMatrix to Eigen::Matrix<double, -1, -1>
inline Eigen::Matrix<int, -1, -1> fn_convert_RMatrix_to_Eigen_int(const RcppParallel::RMatrix<int> &RMat) {
  
  Eigen::Matrix<int, -1, -1> EigenMat(RMat.nrow(), RMat.ncol());
  
  for (int j = 0; j < RMat.ncol(); ++j) { // col-major storage
    for (int i = 0; i < RMat.nrow(); ++i) {
      EigenMat(i, j) = RMat(i, j);
    }
  } 
  
  return EigenMat; 
} 




 
// Function to convert Eigen::MatrixXd (double) to Rcpp::NumericMatrix
inline Rcpp::NumericMatrix fn_convert_EigenMat_to_RcppMat_dbl(const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> EigenMat) {
  
  Rcpp::NumericMatrix RMat(EigenMat.rows(), EigenMat.cols());
  
  for (int j = 0; j < EigenMat.cols(); ++j) { // col-major storage
    for (int i = 0; i < EigenMat.rows(); ++i) {
      RMat(i, j) = EigenMat(i, j);
    }
  }
  
  return RMat;
} 

// Function to convert Eigen::MatrixXi (int) to Rcpp::IntegerMatrix
inline Rcpp::IntegerMatrix fn_convert_EigenMat_to_RcppMat_int(const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> EigenMat) {
  
  Rcpp::IntegerMatrix RMat(EigenMat.rows(), EigenMat.cols());
  
  for (int j = 0; j < EigenMat.cols(); ++j) { // col-major storage
    for (int i = 0; i < EigenMat.rows(); ++i) {
      RMat(i, j) = EigenMat(i, j);
    }
  } 
  
  return RMat;
} 

// Function to convert Eigen::VectorXd (double) to Rcpp::NumericVector
inline Rcpp::NumericVector fn_convert_EigenVec_to_RcppVec_dbl(const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> EigenVec) {
  
  Rcpp::NumericVector RVec(EigenVec.size());
  
  for (int i = 0; i < EigenVec.size(); ++i) {
    RVec(i) = EigenVec(i);
  } 
  
  return RVec;
} 

 
 inline Rcpp::IntegerVector fn_convert_EigenVec_to_RcppVec_int(const Eigen::Ref<const Eigen::Matrix<int, -1, 1>> EigenVec) {
  
  Rcpp::IntegerVector RVec(EigenVec.size());
  
  for (int i = 0; i < EigenVec.size(); ++i) {
    RVec(i) = EigenVec(i); 
  }
  
  return RVec; 
}




 
 
 inline Eigen::Matrix<double, -1, -1> fn_convert_RcppMat_to_EigenMat(const Rcpp::NumericMatrix &RMat) {
  
  Eigen::Matrix<double, -1, -1>  EigenMat(RMat.nrow(), RMat.ncol());
  
  for (int j = 0; j < RMat.ncol(); ++j) { // col-major storage
    for (int i = 0; i < RMat.nrow(); ++i) {
      EigenMat(i, j) = RMat(i, j);
    } 
  }
  
  return EigenMat;
} 

 
 inline Eigen::Matrix<int, -1, -1>  fn_convert_RcppMat_to_EigenMat(const Rcpp::IntegerMatrix &RMat) { 
  
  Eigen::Matrix<int, -1, -1>  EigenMat(RMat.nrow(), RMat.ncol());
  
  for (int j = 0; j < RMat.ncol(); ++j) { // col-major storage
    for (int i = 0; i < RMat.nrow(); ++i) {
      EigenMat(i, j) = RMat(i, j); 
    }
  }
   
  return EigenMat;
}



 
 
 inline Eigen::Matrix<double, -1, 1 > fn_convert_RcppVec_to_EigenVec_dbl(const Rcpp::NumericVector &RVec) {
   
  Eigen::Matrix<double, -1, 1 > EigenVec(RVec.size());
  
  for (int i = 0; i < RVec.size(); ++i) {
    EigenVec(i) = RVec(i);
  }
  
  return EigenVec;
}

 
 inline Eigen::Matrix<int, -1, 1 > fn_convert_RcppVec_to_EigenVec_int(const Rcpp::IntegerVector &RVec) {
  
  Eigen::Matrix<int, -1, 1 > EigenVec(RVec.size());
  
  for (int i = 0; i < RVec.size(); ++i) {
    EigenVec(i) = RVec(i);
  }
  
  return EigenVec;
}









inline RcppParallel::RVector<double> fn_convert_NumericVector_to_RVector(const Rcpp::NumericVector &r_vec) {
  
  RcppParallel::RVector<double> rcpp_parallel_mat(r_vec);
  
  return rcpp_parallel_mat;
  
}


inline RcppParallel::RVector<int> fn_convert_IntegerVector_to_RVector(const Rcpp::IntegerVector &r_vec) {
  
  RcppParallel::RVector<int> rcpp_parallel_mat(r_vec);
  
  return rcpp_parallel_mat;
  
} 



inline RcppParallel::RMatrix<double> fn_convert_NumericMatrix_to_RMatrix(const Rcpp::NumericMatrix &r_matrix) {
  
  RcppParallel::RMatrix<double> r_parallel_matrix(r_matrix);
  
  return r_parallel_matrix;
  
}




inline RcppParallel::RMatrix<int> fn_convert_IntegerMatrix_to_RMatrix(const Rcpp::IntegerMatrix &r_matrix) {
  
  RcppParallel::RMatrix<int> r_parallel_matrix(r_matrix);
  
  return r_parallel_matrix;
  
}










inline tbb::concurrent_vector<RcppParallel::RVector<double>>   fn_convert_std_vec_of_NumericVector_to_tbb_conc_vec_of_RVector(   const std::vector<Rcpp::NumericVector> &std_vec,
                                                                                                                                 tbb::concurrent_vector<RcppParallel::RVector<double>> &tbb_vec) {
  
  for (int i = 0; i < std_vec.size(); ++i) {
    
    const Rcpp::NumericVector &temp_NumericVector = std_vec[i];
    RcppParallel::RVector<double> temp_RVector(temp_NumericVector);
    tbb_vec.emplace_back(temp_RVector);
    
  }
  
  return  tbb_vec;
  
}




inline tbb::concurrent_vector<RcppParallel::RVector<int>>   fn_convert_std_vec_of_IntegerVector_to_tbb_conc_vec_of_RVector(
    const std::vector<Rcpp::IntegerVector> &std_vec,
    tbb::concurrent_vector<RcppParallel::RVector<int>> &tbb_vec) {
  
  
  for (int i = 0; i < std_vec.size(); ++i) {
    
    const Rcpp::IntegerVector &vec = std_vec[i];
    RcppParallel::RVector<int> temp_RVector(vec);
    tbb_vec.emplace_back(temp_RVector);
    
  }
  
  return  tbb_vec;
  
}





inline tbb::concurrent_vector<RcppParallel::RMatrix<double>>   fn_convert_std_vec_of_NumericMatrix_to_tbb_conc_vec_of_RMatrix(
    const std::vector<Rcpp::NumericMatrix> &std_vec,
    tbb::concurrent_vector<RcppParallel::RMatrix<double>> &tbb_vec) {
  
  
  for (int i = 0; i < std_vec.size(); ++i) {
    
    const Rcpp::NumericMatrix &temp_NumericMatrix = std_vec[i];
    RcppParallel::RMatrix<double> temp_RMatrix(temp_NumericMatrix);
    tbb_vec.emplace_back(temp_RMatrix);
    
  }
  
  return  tbb_vec;
  
}




inline tbb::concurrent_vector<RcppParallel::RMatrix<int>>   fn_convert_std_vec_of_IntegerMatrix_to_tbb_conc_vec_of_RMatrix(
    const std::vector<Rcpp::IntegerMatrix> &std_vec,
    tbb::concurrent_vector<RcppParallel::RMatrix<int>> &tbb_vec) {
  
  
  for (int i = 0; i < std_vec.size(); ++i) {
    
    const Rcpp::IntegerMatrix &mat = std_vec[i];
    RcppParallel::RMatrix<int> temp_RMatrix(mat);
    tbb_vec.emplace_back(temp_RMatrix);
    
  }
  
  return  tbb_vec;
  
}




inline Eigen::Matrix<double, -1, -1> fn_convert_RMatrix_to_Eigen_double(const RcppParallel::RMatrix<double> &RMat) {
  
  Eigen::Matrix<double, -1, -1> EigenMat(RMat.nrow(), RMat.ncol());
  
  for (int j = 0; j < RMat.ncol(); ++j) { // col-major storage
    for (int i = 0; i < RMat.nrow(); ++i) {
      EigenMat(i, j) = RMat(i, j);
    }
  }
  
  return EigenMat;
  
}


inline std::deque<bool> fn_convert_RVector_to_deque_vec_bool(const RcppParallel::RVector<int> &RVec) {
  
  std::deque<bool> stdDeque(RVec.size());
  
  for (size_t i = 0; i < RVec.size(); ++i) {
    stdDeque[i] = static_cast<bool>(RVec[i]);
  }
  
  return stdDeque;
  
}




inline std::vector<int> fn_convert_RVector_to_std_vec_Int(const RcppParallel::RVector<int> &RVec) {
  std::vector<int> stdVec(RVec.size());
  
  for (size_t i = 0; i < RVec.size(); ++i) {
    stdVec[i] = RVec[i];
  }
  
  return stdVec;
  
}



inline std::vector<double> fn_convert_RVector_to_std_vec_double(const RcppParallel::RVector<double> &RVec) {
  std::vector<double> stdVec(RVec.size());
  
  for (size_t i = 0; i < RVec.size(); ++i) {
    stdVec[i] = RVec[i];
  }
  
  return stdVec;
  
}



inline std::vector<std::string> fn_convert_string_RVector_to_std_vec(const RcppParallel::RVector<std::string> &RVec) {
  std::vector<std::string> stdVec(RVec.size());
  
  for (size_t i = 0; i < RVec.size(); ++i) {
    stdVec[i] = RVec[i];
  }
  
  return stdVec;
  
}






inline std::vector<Eigen::Matrix<double, -1, 1>> fn_convert_tbb_vec_of_RVector_to_EigenColVec_double(const tbb::concurrent_vector<RcppParallel::RVector<double>> &RVecs) {
  
  std::vector<Eigen::Matrix<double, -1, 1>> EigenColVecs(RVecs.size());
  
  for (size_t i = 0; i < RVecs.size(); ++i) {
    
    const RcppParallel::RVector<double> &RVec = RVecs[i];
    Eigen::Matrix<double, -1, 1> EigenColVec(RVec.size());
    
    for (size_t j = 0; j < RVec.size(); ++j) {
      EigenColVec(j) = RVec[j];
    }
    
    EigenColVecs[i] = EigenColVec;
    
  }
  
  return EigenColVecs;
  
}






inline std::vector<Eigen::Matrix<double, -1, -1>> fn_convert_tbb_vec_of_RMatrix_to_Eigen_double(const tbb::concurrent_vector<RcppParallel::RMatrix<double>> &RMats) {
  
  std::vector<Eigen::Matrix<double, -1, -1>> EigenMats(RMats.size());
  
  for (size_t i = 0; i < RMats.size(); ++i) {
    EigenMats[i] = fn_convert_RMatrix_to_Eigen_double(RMats[i]);
  }
  
  return EigenMats;
  
}





inline std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> fn_convert_tbb_vec_of_vec_of_RMatrix_to_Eigen_double(const tbb::concurrent_vector<tbb::concurrent_vector<RcppParallel::RMatrix<double>>> &RMatVecs) {
  
  std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> EigenMatVecs(RMatVecs.size());
  
  for (size_t i = 0; i < RMatVecs.size(); ++i) {
    EigenMatVecs[i].resize(RMatVecs[i].size());
    
    for (size_t j = 0; j < RMatVecs[i].size(); ++j) {
      EigenMatVecs[i][j] = fn_convert_RMatrix_to_Eigen_double(RMatVecs[i][j]);
    }
  }
  
  return EigenMatVecs;
}







inline std::vector<std::vector<Eigen::Matrix<int, -1, -1>>> fn_convert_tbb_vec_of_vec_of_RMatrix_to_Eigen_int(const tbb::concurrent_vector<tbb::concurrent_vector<RcppParallel::RMatrix<int>>> &RMatVecs) {
  
  std::vector<std::vector<Eigen::Matrix<int, -1, -1>>> EigenMatVecs(RMatVecs.size());
  
  for (size_t i = 0; i < RMatVecs.size(); ++i) {
    EigenMatVecs[i].resize(RMatVecs[i].size());
    
    for (size_t j = 0; j < RMatVecs[i].size(); ++j) {
      EigenMatVecs[i][j] = fn_convert_RMatrix_to_Eigen_int(RMatVecs[i][j]);
    }
  }
  
  return EigenMatVecs;
  
}








inline std::vector<std::vector<Eigen::Matrix<int, -1, 1>>> fn_convert_tbb_vec_of_vec_of_RVector_to_EigenColVec_int(const tbb::concurrent_vector<tbb::concurrent_vector<RcppParallel::RVector<int>>> &RVecs) {
  
  std::vector<std::vector<Eigen::Matrix<int, -1, 1>>> EigenColVecs(RVecs.size());
  
  for (size_t i = 0; i < RVecs.size(); ++i) {
    EigenColVecs[i].resize(RVecs[i].size());
    
    for (size_t j = 0; j < RVecs[i].size(); ++j) {
      const RcppParallel::RVector<int> &RVec = RVecs[i][j];
      Eigen::Matrix<int, -1, 1> EigenColVec(RVec.size());
      
      for (size_t k = 0; k < RVec.size(); ++k) {
        EigenColVec(k) = RVec[k];
      }
      EigenColVecs[i][j] = EigenColVec;
    }
  }
  
  return EigenColVecs;
  
}








inline Eigen::Matrix<double, -1, 1> fn_convert_RVector_to_EigenColVec_double(const RcppParallel::RVector<double> &RVec) {
  
  Eigen::Matrix<double, -1, 1> EigenColVec(RVec.size());
  
  for (size_t i = 0; i < RVec.size(); ++i) {
    EigenColVec(i) = RVec[i];
  }
  
  return EigenColVec;
  
}







 
 


 



 inline tbb::concurrent_vector<std::string> initialize_tbb_string_vector(std::vector<std::string> input_strings) {
  
  tbb::concurrent_vector<std::string> output_vector;
  output_vector.reserve(input_strings.size());
  
  // Emplace each string into the concurrent vector (thread-safe)
  for (const auto& str : input_strings) {
    output_vector.emplace_back(str);
  }
  
  return output_vector;
  
}







inline std::vector<std::string> fn_convert_tbb_vec_to_std_vec_string(tbb::concurrent_vector<std::string> tbbVec) {
  
  std::vector<std::string> stdVec(tbbVec.size());
  
  for (size_t i = 0; i < tbbVec.size(); ++i) {
    stdVec[i] = tbbVec[i];
  }
  
  return stdVec;
  
} 





 

 

// Worker for replicating Rcpp IntegerMatrix in parallel
struct ReplicateRcppMatWorker : public RcppParallel::Worker {
  
  const Rcpp::IntegerMatrix input_matrix;
  std::vector<Rcpp::IntegerMatrix> &output_vector;
  
  /// constructor
  ReplicateRcppMatWorker(const Rcpp::IntegerMatrix &input_matrix_,
                         std::vector<Rcpp::IntegerMatrix> &output_vector_)
    : input_matrix(input_matrix_),
      output_vector(output_vector_) 
    {}
   
  // Parallel operator
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      output_vector[i] = Rcpp::clone(input_matrix); // Use clone to ensure independence
    }
  }
  
};  







// Parallel version of the replicate function
inline std::vector<Rcpp::IntegerMatrix> replicate_Rcpp_Mat_int_parallel(const Rcpp::IntegerMatrix input_matrix,
                                                                 int N) {
  std::vector<Rcpp::IntegerMatrix> output_vector(N);
  
  // Create the worker 
  ReplicateRcppMatWorker worker(input_matrix, output_vector);
   
  // Run parallelFor
  RcppParallel::parallelFor(0, N, worker);
   
  return output_vector;
} 







































