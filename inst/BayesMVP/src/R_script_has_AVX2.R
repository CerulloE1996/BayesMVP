# 
# require(Rcpp)
# 
# Rcpp::sourceCpp("cpu_check.cpp")
# 
# features <- checkCPUFeatures()
# has_avx2 <- features$has_avx2
# writeLines(as.character(as.integer(has_avx2)))
# 



require(Rcpp)

# Rcpp::sourceCpp("cpu_check.cpp")

Rcpp::sourceCpp(code = " 
              #include <Rcpp.h>
              #include <bitset>
              #ifdef _WIN32
              #include <x86intrin.h>
              #else
              #include <cpuid.h>
              #endif 
              
              // [[Rcpp::export]]
              Rcpp::List checkCPUFeatures() {
                
                
                #ifdef _WIN32
                int info[4];
                int subinfo[4];
                
                //// Get basic features
                __cpuid(info, 1);  // Using __cpuid intrinsic instead of inline assembly
                std::bitset<32> f_ecx(info[2]); 
                std::bitset<32> f_edx(info[3]);  // Added EDX bitset for AVX detection
                bool has_avx = f_ecx[28];  // AVX is bit 28 in ECX
                bool has_fma = f_ecx[12];  // FMA is bit 12 in ECX
                
                // Get extended features
                __cpuidex(subinfo, 7, 0);  // Using __cpuidex for extended features
                std::bitset<32> f_ebx(subinfo[1]);
                bool has_avx2 = f_ebx[5];      // AVX2 is bit 5 in EBX
                bool has_avx512 = f_ebx[16];   // AVX512F is bit 16 in EBX
                #else 
                unsigned int eax, ebx, ecx, edx;
                bool has_avx = false;
                bool has_avx2 = false;
                bool has_avx512 = false;
                bool has_fma = false;
                
                // Get basic features
                if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
                  has_avx = (ecx & bit_AVX) != 0;  // AVX is bit 28 in ECX
                  has_fma = (ecx & bit_FMA) != 0;  // Using the proper bit_FMA constant
                } 
                
                // Get extended features
                if (__get_cpuid_max(0, &eax) >= 7) {
                  __cpuid_count(7, 0, eax, ebx, ecx, edx);
                  has_avx2 = (ebx & bit_AVX2) != 0;
                  has_avx512 = (ebx & bit_AVX512F) != 0;
                }
                #endif
                
                // Logical consistency check: if AVX2 or AVX-512 is present, AVX must be present!
                  if ((has_avx2 || has_avx512) && !has_avx) {
                    has_avx = true;  // Force AVX to true if AVX2 or AVX-512 is detected
                  }
                
                int has_AVX_int =     (has_avx == true)     ? 1 : 0;
                int has_AVX2_int =    (has_avx2 == true)    ? 1 : 0;
                int has_AVX_512_int = (has_avx512 == true)  ? 1 : 0;
                int has_FMA_int =     (has_fma == true)     ? 1 : 0;
                
                return Rcpp::List::create(
                    has_AVX_int,
                    has_AVX2_int,
                    has_AVX_512_int,
                    has_FMA_int
                );
                
              }")


features <- checkCPUFeatures()
##
# has_AVX     <- features[[1]]
has_AVX2    <- features[[2]]
# has_AVX_512 <- features[[3]]
# features <- checkCPUFeatures()
# has_AVX  <- features$has_AVX
writeLines(as.character(as.integer(has_AVX2)))








