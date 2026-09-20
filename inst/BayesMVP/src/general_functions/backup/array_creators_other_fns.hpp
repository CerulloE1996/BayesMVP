
#pragma once

 

#include <Rcpp.h>


 


ALWAYS_INLINE  std::vector<Rcpp::NumericMatrix> vec_of_mats_Rcpp(int n_rows, int n_cols, int n_matrices) {
  
  std::vector<Rcpp::NumericMatrix> vec_of_mats(n_matrices);
  
  for (int i = 0; i < n_matrices; ++i) {
    vec_of_mats[i] = Rcpp::NumericMatrix(n_rows, n_cols);
  } 
  
  return vec_of_mats;
  
} 









ALWAYS_INLINE  int flatten_index(int i, int j, int dim1) {
  return i + j * dim1;  // convert (i, j) into the ROW index for the 2D matrix
}



ALWAYS_INLINE  Rcpp::NumericMatrix create_3D_array_as_2D_RcppNumericMat(const int dim1,
                                                                const int dim2,
                                                                const int dim3) {
  
  Rcpp::NumericMatrix array_2D(dim1 * dim2, dim3);
  
  // Fill the array with values
  for (int k = 0; k < dim3; ++k) {  // loop through columns first since col-major storage (so more efficient)
    for (int j = 0; j < dim2; ++j) {
      for (int i = 0; i < dim1; ++i) {
        int row_index = flatten_index(i, j, dim1); // flatten the (i, j)'th index
        array_2D(row_index, k) = 0;  // fill with zero's
      }
    }
  }
  
  return array_2D;
  
}





ALWAYS_INLINE std::vector<Rcpp::NumericMatrix> create_3D_array_as_vector_of_2D_matrices(const Rcpp::NumericMatrix& array_2D,
                                                                          const int dim1,
                                                                          const int dim2,
                                                                          const int dim3) {
  // Create a vector to hold dim3 2D matrices (each of size dim1 x dim2)
  std::vector<Rcpp::NumericMatrix> array_3D(dim3, Rcpp::NumericMatrix(dim1, dim2));
  
  // Fill the vector of 2D matrices with values from the 2D input matrix
  for (int k = 0; k < dim3; ++k) {  // Iterate over the 3rd dimension (depth) 
    for (int j = 0; j < dim2; ++j) {  // Iterate over the 2nd dimension (columns)
      for (int i = 0; i < dim1; ++i) {  // Iterate over the 1st dimension (rows)
        int row_index = i + j * dim1;  // Flatten the (i, j) index in the input 2D matrix
        array_3D[k](i, j) = array_2D(row_index, k);  // Assign the value to the correct matrix slice
      }
    } 
  }
  
  return array_3D;
} 


 