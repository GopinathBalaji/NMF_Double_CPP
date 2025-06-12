#define EIGEN_USE_BLAS
//#define EIGEN_RUNTIME_NO_MALLOC

#include <iostream>
#include <fstream>
#include <random>
#include <cstdlib>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <map>
#include <matio.h>

void saveMatrixVector(const std::string& filename, const std::vector<Eigen::MatrixXd>& matrices) {
    std::ofstream file(filename, std::ios::binary);
	// Write the number of matrices
    size_t number_of_matrices = matrices.size();
    file.write(reinterpret_cast<char*>(&number_of_matrices), sizeof(size_t));
    // Write each matrix
    for (const auto& matrix : matrices) {
        typename Eigen::MatrixXd::Index rows = matrix.rows();
        typename Eigen::MatrixXd::Index cols = matrix.cols();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(typename Eigen::MatrixXd::Index));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(typename Eigen::MatrixXd::Index));
        file.write(reinterpret_cast<const char*>(matrix.data()), rows * cols * sizeof(double));
    }
    file.close();
}

void saveMatrix(const std::string& filename, const Eigen::MatrixXd& matrix) {
    std::ofstream file(filename, std::ios::binary);
    Eigen::MatrixXd::Index rows=matrix.rows(), cols=matrix.cols();
    file.write((char*) (&rows), sizeof(Eigen::MatrixXd::Index));
    file.write((char*) (&cols), sizeof(Eigen::MatrixXd::Index));
    file.write((char*) matrix.data(), rows*cols*sizeof(Eigen::MatrixXd::Scalar) );
    file.close();
}

void saveMatrixVectorAsText(const std::string& filename, const std::vector<Eigen::MatrixXd>& matrices) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing." << std::endl;
        return;
    }
    // Write the number of matrices
    size_t number_of_matrices = matrices.size();
    file << number_of_matrices << std::endl;
    // Write each matrix
    for (const auto& matrix : matrices) {
        // Write dimensions of the matrix
        Eigen::MatrixXd::Index rows = matrix.rows();
        Eigen::MatrixXd::Index cols = matrix.cols();
        file << rows << " " << cols << std::endl;
        // Write matrix data
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                file << matrix(i, j);
                if (j + 1 != cols) file << " "; // Space between columns, no extra space at line end
            }
            file << std::endl;  // Newline after each row
        }
    }
    file.close();
}

void saveMatrixAsText(const std::string& filename, const Eigen::MatrixXd& matrix) {
    std::ofstream file(filename);  // Default is text mode, not binary
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing." << std::endl;
        return;
    }
    // Get the dimensions of the matrix
    Eigen::MatrixXd::Index rows = matrix.rows(), cols = matrix.cols();
    // Write dimensions at the top of the file (optional, can remove if not wanted)
    file << rows << " " << cols << std::endl;
    // Write matrix elements, each row on a new line and elements separated by spaces
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << matrix(i, j);
            if (j + 1 != cols) file << " "; // Space between columns, no extra space at line end
        }
        file << std::endl;  // Newline after each row
    }
    file.close();
}

Eigen::MatrixXd readMatFile(const std::string& filename, const std::string& varName) {
	mat_t *matfp = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
	if (matfp == nullptr) {
		throw std::runtime_error("Error opening MAT file");
	}
	// Read the variable from the file
	matvar_t *matvar = Mat_VarRead(matfp, varName.c_str());
	if (matvar == nullptr) {
		Mat_Close(matfp);
		throw std::runtime_error("Error reading the variable from MAT file");
	}
	// Check if the variable is of the expected type and dimensions
	std::cout << matvar->class_type<<std::endl;
	//std::cout << MAT_C_DOUBLE << " " << MAT_C_SINGLE << " " << MAT_C_UINT16 << " " << MAT_C_INT16 << std::endl;
	if (matvar->class_type != MAT_C_SINGLE || matvar->rank != 2) {
		Mat_VarFree(matvar);
		Mat_Close(matfp);
		throw std::runtime_error("Variable is not a 2D float matrix");
	}
	// Transfer the data to an Eigen matrix
	size_t rows = matvar->dims[0];
	size_t cols = matvar->dims[1];
	Eigen::MatrixXf mat2(rows, cols);
	memcpy(mat2.data(), matvar->data, rows * cols * sizeof(float));

	Eigen::MatrixXd mat = mat2.cast<double>();
	// Clean up
	Mat_VarFree(matvar);
	Mat_Close(matfp);
	return mat;
}

//return U_final,V_final,nIter_final,elapse_final,bSuccess,objhistory_final
//def NMF(X=None, k=None, options=None, bSuccess=None, U_=None, V_=None):
/*
std::map<std::string, double>& bSuccess, const Eigen::MatrixXd& U_, 
const Eigen::MatrixXd& V_) {

	int mFea = X.rows();
	int nSmp = X.cols();
	Eigen::MatrixXd U;
	Eigen::MatrixXd V;

    if (U_.rows() == 0) {
        U = Eigen::MatrixXd::Random(mFea, k).cwiseAbs(); // Random numbers between -1 and 1, made positive
        Eigen::VectorXd norms = U.rowwise().squaredNorm();
        for (int i = 0; i < norms.size(); ++i) {
            norms(i) = std::max(norms(i), 1e-10); // Avoid division by zero
        }
        U = U.array().rowwise() / norms.transpose().array();
        std::cout << "Entering NMF, U_ is null" << std::endl;

        if (V_.rows() == 0) {
            V = Eigen::MatrixXd::Random(nSmp, k).cwiseAbs();
            V = V.array().rowwise() / V.colwise().sum().array();
            std::cout << "Entering NMF, U_ is null, V_ is null" << std::endl;
        } else {
            V = V_;
        }
    } else {
        U = U_;
        if (V_.rows() == 0) {
            V = Eigen::MatrixXd::Random(nSmp, k).cwiseAbs();
            V = V.array().rowwise() / V.colwise().sum().array();
            std::cout << "Entering NMF, V_ is null" << std::endl;
        } else {
            V = V_;
        }
    }
}
*/

double CalculateObj(const Eigen::MatrixXd& X, const Eigen::MatrixXd& U, const Eigen::MatrixXd& V, bool deltaVU = false, bool dVordU = true) {
    Eigen::MatrixXd dX, dV;
    const int maxM = 62500000;
    int mFea = X.rows(), nSmp = X.cols();
    int mn = X.size();
    int nBlock = std::floor(mn * 3.0 / maxM);
    double obj_NMF = 0;

    if (mn < maxM) {
        dX = U * V.transpose() - X;
        obj_NMF = dX.squaredNorm();
        if (deltaVU) {
            if (dVordU) {
                dV = dX.transpose() * U;
            } else {
                dV = dX * V;
            }
        }
    } else {
        if (deltaVU) {
            if (dVordU) {
                dV = Eigen::MatrixXd::Zero(V.rows(), V.cols());
            } else {
                dV = Eigen::MatrixXd::Zero(U.rows(), U.cols());
            }
        }
        int iter1 = std::ceil(nSmp / int(nBlock)) + 1;
        for (int i = 1; i < iter1; ++i) {
            int startIndex = (i - 1) * nBlock;
            int endIndex = (i == iter1 - 1) ? nSmp : i * nBlock;

            Eigen::MatrixXd VBlock = V.block(startIndex, 0, endIndex - startIndex, V.cols());
            dX = U * VBlock.transpose() - X.block(0, startIndex, X.rows(), endIndex - startIndex);
            obj_NMF += dX.squaredNorm();
            if (deltaVU) {
                if (dVordU) {
                    dV.block(startIndex, 0, endIndex - startIndex, dV.cols()) = dX.transpose() * U;
                } else {
					std::cout << "Should never visit here" << std::endl;
                    // Adjusting for dU computation would require a different setup, not shown here
                    // This part of the original Python code is not fully clear (`dU` is not previously defined)
                }
            }
        }
    }

    return obj_NMF;
}

//std::pair<Eigen::MatrixXd, Eigen::MatrixXd> NormalizeUV(Eigen::MatrixXd U, Eigen::MatrixXd V, bool NormV, int Norm) {
void NormalizeUV(Eigen::MatrixXd& U, Eigen::MatrixXd& V, bool NormV, int Norm) {
    int nSmp = V.rows();
    int mFea = U.rows();

    if (Norm == 2) {
        Eigen::VectorXd norms;
        if (NormV) {
			norms = V.colwise().squaredNorm().cwiseSqrt().cwiseMax(1e-10);
			for (int i = 0; i < V.rows(); ++i) {
				V.row(i).array() /= norms.transpose().array();
			}
			for (int i = 0; i < U.rows(); ++i) {
				U.row(i).array() *= norms.transpose().array();
			}

			//V = V.array() / norms.replicate(1, nSmp).transpose().array(); //key
			//U = U.array() * norms.replicate(1, mFea).transpose().array(); //key
        } else {
			norms = U.colwise().squaredNorm().cwiseSqrt().cwiseMax(1e-10);
			for (int i = 0; i < U.rows(); ++i) {
				U.row(i).array() /= norms.transpose().array();
			}
			for (int i = 0; i < V.rows(); ++i) {
				V.row(i).array() *= norms.transpose().array();
			}

			//U = U.array() / norms.replicate(1, mFea).transpose().array(); //key
			//V = V.array() * norms.replicate(1, nSmp).transpose().array(); //key
        }
    } else {
        Eigen::VectorXd norms;
        if (NormV) {
            norms = V.cwiseAbs().colwise().sum().cwiseMax(1e-10);
			for (int i = 0; i < V.rows(); ++i) {
				V.row(i).array() /= norms.transpose().array();
			}
			for (int i = 0; i < U.rows(); ++i) {
				U.row(i).array() *= norms.transpose().array();
			}
			//V = V.array() / norms.replicate(1, nSmp).transpose().array(); //key
			//U = U.array() * norms.replicate(1, mFea).transpose().array(); //key
        } else {
            norms = U.cwiseAbs().colwise().sum().cwiseMax(1e-10);
			for (int i = 0; i < U.rows(); ++i) {
				U.row(i).array() /= norms.transpose().array();
			}
			for (int i = 0; i < V.rows(); ++i) {
				V.row(i).array() *= norms.transpose().array();
			}
			//U = U.array() / norms.replicate(1, mFea).transpose().array(); //key
			//V = V.array() * norms.replicate(1, nSmp).transpose().array(); //key
        }
    }
    //return {U, V};
}

void Normalize(Eigen::MatrixXd& U, Eigen::MatrixXd& V){
	NormalizeUV(U,V,0,1);
}


void NMF(const Eigen::MatrixXd& X, int k, std::map<std::string, int>& bSuccess, Eigen::MatrixXd& U_, Eigen::MatrixXd& V_, 
Eigen::MatrixXd& res_U, Eigen::MatrixXd& res_V,  int& res_nIter, std::vector<double>& res_objhistory, int nRepeat, int minIterOrig){  //nRepeat=30, minIterOrig=50

//void NMF(){
//void updateMatrices(Eigen::MatrixXd& X, Eigen::MatrixXd& U, Eigen::MatrixXd& V, const double differror, const int minIter, const double meanFitRatio, std::vector<double>& objhistory, bool selectInit, int maxIter, const std::map<std::string, bool>& options) {
//parameter: Eigen::MatrixXd& X, Eigen::MatrixXd& U, Eigen::MatrixXd& V

    //double meanFit = 0.0; // Initialize meanFit if used outside this snippet
/*
options["maxIter"] = 200;
options["error"] = 1e-6;
options["nRepeat"] = 30;
options["minIter"] = 50;
options["meanFitRatio"] = 0.1;
options["rounds"] = 30;
options["alpha"] = [0.01, 0.01]
Rounds = options["rounds"]
*/
	int maxIter = 200;
	double error = 1e-6;
	//int nRepeat = 30;   //important============, moved to function parameter
	int nIter = 0;
	double maxErr = 0;
	int nStepTrial = 0;
	//int minIterOrig = 50; //important===========
	double differror = 1e-6;
	int minIter = minIterOrig - 1;

	bSuccess["bSuccess"] = 1;
	int mFea = X.rows();
	int nSmp = X.cols();
	bool NormV = false;
	int Norm = 1;
	
	double meanFitRatio = 0.1;
	double meanFit;
	bool selectInit = true;

	//std::map<std::string, int> bSuccess;	
	std::vector<double> objhistory;
	std::vector<double> objhistory_final;
	Eigen::MatrixXd U;
	Eigen::MatrixXd V;

	std::random_device rd;
    std::mt19937 gen(rd());  
	std::uniform_real_distribution<double> dis(-1.0, 1.0);


	std::cout<<"starts here"<<std::endl;
	if (U_.rows() == 0) {
		U = Eigen::MatrixXd::NullaryExpr(mFea, k, [&](){return dis(gen);}).cwiseAbs();
		//U = Eigen::MatrixXd::Random(mFea, k).cwiseAbs();

		std::cout<<"enter here 1"<<std::endl;
		Eigen::VectorXd norms = U.colwise().squaredNorm();
		norms = norms.cwiseSqrt();
		norms = norms.cwiseMax(1e-10);
		//std::cout<<"enter here 1b"<<std::endl;
		//std::cout<<U.rows() << " " << U.cols() <<std::endl;
		//std::cout<<norms.rows() << " " << norms.cols() <<std::endl;
		//std::cout<<norms.replicate(1, mFea).transpose().rows() << " " << norms.replicate(1, mFea).transpose().cols() << std::endl;
		U = U.array() / norms.replicate(1, mFea).transpose().array(); //key
		std::cout << "Entering NMF, U_ is null\n";

		if (V_.rows() == 0) {
			V = Eigen::MatrixXd::NullaryExpr(nSmp, k, [&](){return dis(gen);}).cwiseAbs();
			//V = Eigen::MatrixXd::Random(nSmp, k).cwiseAbs();
			V = V.array() / V.sum(); //key
			std::cout << "Entering NMF, U_ is null, V_ is null\n";
		}else{
			V = V_;
		}
	}else{
		U = U_;
		if (V_.rows() == 0) {
			V = Eigen::MatrixXd::NullaryExpr(nSmp, k, [&](){return dis(gen);}).cwiseAbs();
			//V = Eigen::MatrixXd::Random(nSmp, k).cwiseAbs();
			V = V.array() / V.sum();
			std::cout << "Entering NMF, V_ is null\n";
		}else{
			V = V_;
		}
	}

	NormalizeUV(U, V, NormV, Norm); //U and V is updated

	int tryNo = 0;
	Eigen::MatrixXd U_final;
	Eigen::MatrixXd V_final;
	int nIter_final; 

	while (tryNo < nRepeat){
		tryNo+=1;
		nIter = 0;
		maxErr = 1.0;
		nStepTrial = 0;
		std::cout << "NMF tryNo" << tryNo << std::endl;
		
		while (maxErr > differror) {
			//Eigen::MatrixXd UT = U.transpose();
			//Eigen::MatrixXd VT = V.transpose();
			Eigen::MatrixXd denominatorU = (V * (U.transpose() * U)).cwiseMax(1e-10);
			V.array() *= (X.transpose() * U).array() / denominatorU.array();

			Eigen::MatrixXd denominatorV = (U * (V.transpose() * V)).cwiseMax(1e-10);
			U.array() *= (X * V).array() / denominatorV.array();

			/*
			Eigen::MatrixXd numeratorU = X.transpose() * U;
			Eigen::MatrixXd denominatorU = V * (U.transpose() * U);
			denominatorU = denominatorU.cwiseMax(1e-10);
			V = V.array() * (numeratorU.array() / denominatorU.array());

			Eigen::MatrixXd numeratorV = X * V;
			Eigen::MatrixXd denominatorV = U * (V.transpose() * V);
			denominatorV = denominatorV.cwiseMax(1e-10);
			U = U.array() * (numeratorV.array() / denominatorV.array());
			*/

			nIter += 1;
			//std::cout << "nIter " << nIter << " minIter " << minIter <<std::endl;
			if (nIter > minIter) {
				if(selectInit){
					std::cout << "selectInit: true " << nIter << " " << meanFit << std::endl;
					double newobj = CalculateObj(X, U, V); // Simplified logic
					objhistory.push_back(newobj); // Assuming objhistory is a std::vector<double>
					meanFit = meanFitRatio * meanFit + (1 - meanFitRatio) * newobj;	
					maxErr = 0; // or another logic to set maxErr
				}
				else{
					/*
					if(maxIter==0){
						double newobj = CalculateObj(X, U, V);
						objhistory.push_back(newobj); // Assuming objhistory is a std::vector<double>
						meanFit = meanFitRatio * meanFit + (1 - meanFitRatio) * newobj;	
					}*/
					if(maxIter>0){
						maxErr = 1.0;
						if(nIter>=maxIter){
							maxErr = 0;
							objhistory.push_back(0);
						}
					}
				}
			}

		}
		if (tryNo == 1) {
			U_final = U;
			V_final = V;
			nIter_final = nIter;
			//elapse_final = elapse;
			objhistory_final = objhistory; // Assuming objhistory is a std::vector<double>
			bSuccess["nStepTrial"] = nStepTrial;
		} else {
			// The comparison logic here assumes objhistory is always a vector in C++
			if (!objhistory.empty() && !objhistory_final.empty() && (objhistory.back() < objhistory_final.back())) {
				U_final = U;
				V_final = V;
				nIter_final = nIter;
				objhistory_final = objhistory;
				bSuccess["nStepTrial"] = nStepTrial;
				/*
				if (selectInit) {
					elapse_final = elapse;
				}else{
					elapse_final += elapse;
				}*/
			}
		}

		if (selectInit){
			if(tryNo < nRepeat) {
				if (U_.rows() == 0) { // Equivalent to checking the shape in Python
					U = Eigen::MatrixXd::NullaryExpr(mFea, k, [&](){return dis(gen);}).cwiseAbs();
					//U = Eigen::MatrixXd::Random(mFea, k).cwiseAbs(); // Random initialization

					Eigen::VectorXd norms = U.colwise().squaredNorm();
					norms = norms.cwiseSqrt();
					//Eigen::VectorXd norms = U.rowwise().squaredNorm().sqrt(); // Compute norms
					norms = norms.cwiseMax(1e-10); // Ensure no division by zero
					//mat = mat.array() / norms.replicate(1, mFea).transpose().array();
					U = U.array() / norms.replicate(1, mFea).transpose().array();
					//U = U.array().colwise() / norms.array();
					//U.array().rowwise() /= norms.transpose().array(); // Normalize
					std::cout << "Entering NMF, selectInit is true, U_ is null\n";
					if (V_.rows() == 0) {
						//V = Eigen::MatrixXd::Random(nSmp, k).cwiseAbs(); // Random initialization
						V = Eigen::MatrixXd::NullaryExpr(nSmp, k, [&](){return dis(gen);}).cwiseAbs();
						V = V.array() / V.sum();
						std::cout << "Entering NMF, selectInit is true, U_ is null, V_ is null\n";
					} else {
						V = V_;
					}
				} else {
					U = U_;
					if (V_.rows() == 0) {
						//V = Eigen::MatrixXd::Random(nSmp, k).cwiseAbs(); // Random initialization
						V = Eigen::MatrixXd::NullaryExpr(nSmp, k, [&](){return dis(gen);}).cwiseAbs();
						V = V.array() / V.sum();
						//V.array().rowwise() /= V.colwise().sum().array();
						//V = V.array().rowwise() / V.colwise().sum().array();
						std::cout << "Entering NMF, selectInit is true, V_ is null\n";
					} else {
						V = V_;
					}
				}
				NormalizeUV(U, V, NormV, Norm); // Assuming NormalizeUV is implemented
			} else {
				//if (tryNo >= nRepeat) {
				tryNo -= 1;
				minIter = 0;
				selectInit = false;
				U = U_final; // Assuming U_final is defined elsewhere
				V = V_final; // Assuming V_final is defined elsewhere
				// Assuming objhistory_final is a std::vector<double>
				objhistory = objhistory_final;
				meanFit = objhistory_final.back() * 10; // Example calculation
				// Additional logic as needed
			}
		}
	}
	nIter_final = nIter_final + minIterOrig;
	Normalize(U_final,V_final);

	//Eigen::MatrixXd& res_U, Eigen::MatrixXd& res_V, 
	//int& res_nIter, std::vector<double>& res_objhistory){
	res_nIter = nIter_final;
	res_U = U_final;
	res_V = V_final;
	res_objhistory = objhistory;
}





double CalculateObjPerView(const Eigen::MatrixXd& X, const Eigen::MatrixXd& U, const Eigen::MatrixXd& V, 
const Eigen::MatrixXd& L, double& alpha, bool deltaVU = false, bool dVordU = true) {
//def CalculateObj(X = None,U = None,V = None,L = None,alpha = None,deltaVU = None,dVordU = None):

	Eigen::MatrixXd dX, dV, tmp;
	const int maxM = 62500000;
    int mFea = X.rows(), nSmp = X.cols();
	int mn = X.size();
	int nBlock = std::floor(mn * 3.0 / maxM);
	double obj_NMF = 0;
	
 	/*
	if deltaVU is None:
		deltaVU = 0
	if dVordU is None:
		dVordU = 1
	*/

	/*
	dV = None
	maxM = 62500000
	mFea,nSmp = X.shape
	mn = np.asarray(X).size
	nBlock = int(np.floor(mn * 3 / maxM))
	obj_NMF = None
	*/
	if(mn < maxM){
		dX = U * V.transpose() - X;
		obj_NMF = dX.squaredNorm();
        if (deltaVU) {
            if (dVordU) {
                dV = dX.transpose() * U + L * V;
            } else {
                dV = dX * V;
            }
        }

		/*
		dX = np.matmul(U, np.transpose(V)) - X
		obj_NMF = np.sum(np.square(dX))
		if deltaVU:
			if dVordU:
				dV = np.matmul(np.transpose(dX), U) + np.matmul(L, V)
			else:
				dV = np.matmul(dX, V)
		*/
	}else{
		//obj_NMF = 0
        if (deltaVU) {
            if (dVordU) {
                dV = Eigen::MatrixXd::Zero(V.rows(), V.cols());
            } else {
                dV = Eigen::MatrixXd::Zero(U.rows(), U.cols());
            }
        }

        int iter1 = std::ceil(nSmp / int(nBlock)) + 1;
        for (int i = 1; i < iter1; ++i) {
            int startIndex = (i - 1) * nBlock;
            int endIndex = (i == iter1 - 1) ? nSmp : i * nBlock;

            Eigen::MatrixXd VBlock = V.block(startIndex, 0, endIndex - startIndex, V.cols());
            dX = U * VBlock.transpose() - X.block(0, startIndex, X.rows(), endIndex - startIndex);
            obj_NMF += dX.squaredNorm();
            if (deltaVU) {
                if (dVordU) {
                    dV.block(startIndex, 0, endIndex - startIndex, dV.cols()) = dX.transpose() * U;
                } else {
					//dV = dU + dX * V.block(startIndex, 0, endIndex - startIndex, V.cols());
					std::cout << "Should never visit here PerView" << std::endl;
                    // Adjusting for dU computation would require a different setup, not shown here
                    // This part of the original Python code is not fully clear (`dU` is not previously defined)
                }
            }
        }

		if(deltaVU){
			if(dVordU){
				dV = dV + L * V;
			}
		}

	}

	tmp = V - L;

	double obj_Lap = tmp.squaredNorm();
	//obj_Lap = np.sum(np.square(tmp))
	dX = U * V.transpose() - X;
	//dX = np.matmul(U, np.transpose(V)) - X
	obj_NMF = dX.squaredNorm();
	//obj_NMF = np.sum(np.square(dX))

	double obj = obj_NMF + alpha * obj_Lap;
	//obj = obj_NMF + alpha * obj_Lap
	return obj;
}


void NormalizeUVPerView(Eigen::MatrixXd& U, Eigen::MatrixXd& V, bool NormV, int Norm) {
    int nSmp = V.rows();
    int mFea = U.rows();
    if (Norm == 2) {
        Eigen::VectorXd norms;
        if (NormV) {
			norms = V.colwise().squaredNorm().cwiseSqrt();
			norms = norms.cwiseMax(1e-10);
			V = V.array() / norms.replicate(1, nSmp).transpose().array(); //key
			U = U.array() * norms.replicate(1, mFea).transpose().array(); //key
        } else {
			norms = U.colwise().squaredNorm().cwiseSqrt();
			norms = norms.cwiseMax(1e-10);
			U = U.array() / norms.replicate(1, mFea).transpose().array(); //key
			V = V.array() * norms.replicate(1, nSmp).transpose().array(); //key
        }
    } else {
        Eigen::VectorXd norms;
        if (NormV) {
            norms = V.cwiseAbs().colwise().sum();
			norms = norms.cwiseMax(1e-10);
			V = V.array() / norms.replicate(1, nSmp).transpose().array(); //key
			U = U.array() * norms.replicate(1, mFea).transpose().array(); //key
        } else {
            norms = U.cwiseAbs().colwise().sum();
			norms = norms.cwiseMax(1e-10);
			U = U.array() / norms.replicate(1, mFea).transpose().array(); //key
			V = V.array() * norms.replicate(1, nSmp).transpose().array(); //key
        }
    }
	//return U, V
	//#return U_final,V_final,nIter_final,elapse_final,bSuccess,objhistory_final
}

void NormalizePerView(Eigen::MatrixXd& U, Eigen::MatrixXd& V){
	NormalizeUVPerView(U,V,0,1);
}


void PerViewNMF(const Eigen::MatrixXd& X, int k, Eigen::MatrixXd& Vo, Eigen::MatrixXd& U, Eigen::MatrixXd& V, 
Eigen::MatrixXd& res_U, Eigen::MatrixXd& res_V, int& res_nIter, std::vector<double>& res_objhistory, int nRepeat, int minIterOrig, int maxIter, double alpha){ //default alpha=0.01
//void PerViewNMF(X = None,k = None,Vo = None,options = None,U = None,V = None){
	/*
	differror = options["error"];
	maxIter = options["maxIter"];
	nRepeat = options["nRepeat"];
	minIterOrig = options["minIter"];
	minIter = minIterOrig - 1;
	meanFitRatio = options["meanFitRatio"];

	bSuccess = {}	
	#differror = options.error
	#maxIter = options.maxIter
	#nRepeat = options.nRepeat
	#minIterOrig = options.minIter
	#minIter = minIterOrig - 1
	#meanFitRatio = options.meanFitRatio
	#alpha = options.alpha
	alpha = options["alpha"]
	bSuccess["bSuccess"] = 1
	*/
	std::random_device rd;
    std::mt19937 gen(rd());  
	std::uniform_real_distribution<double> dis(-1.0, 1.0);

	double differror = 1e-6;
	//int maxIter = 200; //moved to function parameter =================
	//int nRepeat = 30; //moved to function parameter ================
	//int minIterOrig = 50; //moved to function parameter ================
	int minIter = minIterOrig - 1;
	double meanFitRatio = 0.1;

	std::map<std::string, int> bSuccess;
	bSuccess["bSuccess"] = 1;
	//std::vector<float> alpha(2);
	//alpha[0] = 0.01;
	//alpha[1] = 0.01;
	//double alpha = 0.01; //disabled here Qian	

	double error = 1e-6;
	int nIter = 0;
	double maxErr = 0;
	int nStepTrial = 0;

	int mFea = X.rows();
	int nSmp = X.cols();
	bool NormV = false;
	int Norm = 1;
	bool selectInit = true;
	std::vector<double> objhistory;
	std::vector<double> objhistory_final;
	double meanFit;
	
	Eigen::MatrixXd norms;

	//Norm = 1
	//NormV = 0
	//mFea,nSmp = X.shape
	//selectInit = 1
	//norms = None

	if (U.rows() == 0) {
		U = Eigen::MatrixXd::NullaryExpr(mFea, k, [&](){return dis(gen);}).cwiseAbs();
		V = Eigen::MatrixXd::NullaryExpr(nSmp, k, [&](){return dis(gen);}).cwiseAbs();
		//U = Eigen::MatrixXd::Random(mFea, k).cwiseAbs();
		//V = Eigen::MatrixXd::Random(nSmp, k).cwiseAbs();
		//print("Entering PerViewNMF, U is null")
		std::cout << "Entering PerViewNMF, U is null\n";
	} else{
		nRepeat = 1;
	}

	//U,V = Normalize(U,V)
	NormalizePerView(U, V);

	if (nRepeat==1){
		selectInit = false;
		minIterOrig = 0;
		minIter = 0;
		//not translated because never run
		/*
		if maxIter==None or maxIter==0: #might not be run===========
			objhistory = CalculateObj(X=X,U=U,V=V,L=Vo,alpha=alpha)
			meanFit = objhistory * 10
		else:
			#if isfield(options,'Converge') and options.Converge:
			if "Converge" in options and options["Converge"]:
				objhistory = CalculateObj(X=X,U=U,V=V,L=Vo,alpha=alpha)
		*/	
	}else{
		//#if isfield(options,'Converge') and options.Converge:
		//if "Converge" in options and options["Converge"]:
		//	raise Exception('Not implemented!')
	}

	int tryNo = 0;

	Eigen::MatrixXd XU, UU, VUU, XV, VV, UVV, VV_, VVo;
	Eigen::VectorXd tmp_VV_, tmp;
	Eigen::MatrixXd U_final;
	Eigen::MatrixXd V_final;
	int nIter_final; 

	//std::cout << "Arrive here selectInit" << selectInit << "\n";

	while(tryNo < nRepeat){
		tryNo++;
		nIter = 0;
		maxErr = 1;
		nStepTrial = 0;
		//X = np.array(X, dtype=np.float32)
		//U = np.array(U, dtype=np.float32)
		//V = np.array(V, dtype=np.float32)
		//void PerViewNMF(X = None,k = None,Vo = None,options = None,U = None,V = None){

		while (maxErr > differror){

			//std::cout << "Here 1\n";
			XU = X.transpose() * U;
			//std::cout << "Here 2\n";
			UU = U.transpose() * U;
			//std::cout << "Here 3\n";
			VUU = V * UU;
			//std::cout << "Here 4\n";
			XU = XU.array() + alpha * Vo.array();
			//std::cout << "Here 5\n";
			VUU = VUU.array() + alpha * V.array();
			//std::cout << "Here 6\n";
			V = V.array() * (XU.array() / VUU.cwiseMax(1e-10).array());
			
			//std::cout << "Here 7\n";
			XV = X * V;
			//std::cout << "Here 8\n";
			VV = V.transpose() * V;
			//std::cout << "Here 9\n";
			UVV = U * VV;
			//np.divide(U, np.matlib.repmat(norms, 2, 1))
			//std::cout << "Here 10\n";
			tmp_VV_ = VV.diagonal().transpose().array() * U.colwise().sum().array();
			//std::cout << "Here 11\n";
			VV_ = tmp_VV_.replicate(1, mFea).transpose();
			//std::cout << "Here 12\n";
			tmp = (V.array() * Vo.array()).colwise().sum();
			//std::cout << "Here 13\n";
			VVo = tmp.replicate(1, mFea).transpose();
			//std::cout << "Here 14\n";
			XV = XV.array() + alpha * VVo.array();
			//std::cout << "Here 15\n";
			UVV = UVV.array() + alpha * VV_.array();
			//std::cout << "Here 16\n";
			U = U.array() * (XV.array() / UVV.cwiseMax(1e-10).array());
			//std::cout << "Here 17\n";
			NormalizePerView(U, V);

			//std::cout << "Here 18\n";
			nIter++;


			/*
			XU = np.matmul(np.transpose(X), U)
			UU = np.matmul(np.transpose(U), U)
			VUU = np.matmul(V, UU)
			XU = XU + np.multiply(alpha, Vo)
			VUU = VUU + np.multiply(alpha, V)
			V = np.multiply(V, np.divide(XU, np.maximum(VUU,1e-10)))
			# ===================== update U ========================
			XV = np.matmul(X, V)
			VV = np.matmul(np.transpose(V), V)
			UVV = np.matmul(U, VV)
			VV_ = np.matlib.repmat(np.multiply(np.transpose(np.diag(VV)),np.sum(U, axis=0)),mFea,1)
			tmp = np.sum(np.multiply(V,Vo), 0)
			VVo = np.matlib.repmat(tmp,mFea,1)
			XV = XV + np.multiply(alpha, VVo)
			UVV = UVV + np.multiply(alpha, VV_)
			U = np.multiply(U, np.divide(XV, np.maximum(UVV,1e-10)))
			U,V = Normalize(U,V)
			nIter = nIter + 1
			*/
			if (nIter > minIter){
				if(selectInit){ //never reached here because selectInit is false
					std::cout << "Here 19\n";
					double newobj = CalculateObjPerView(X, U, V, Vo, alpha);
					std::cout << "Here 20\n";
					objhistory.push_back(newobj);
					//objhistory = CalculateObj(X=X,U=U,V=V,L=Vo,alpha=alpha)
					maxErr = 0;
				} else{
					//if len(maxIter)==0:
					/*
					if maxIter==None or maxIter==0: #might not be run===========
						newobj = CalculateObj(X=X,U=U,V=V,L=Vo,alpha=alpha)
						objhistory = np.array([objhistory,newobj])
						meanFit = meanFitRatio * meanFit + (1 - meanFitRatio) * newobj
						maxErr = (meanFit - newobj) / meanFit
					else:
					*/
						/*
						#if isfield(options,'Converge') and options.Converge:
						if "Converge" in options and options["Converge"]:
							newobj = CalculateObj(X=X,U=U,V=V,L=Vo,alpha=alpha)
							objhistory = np.array([objhistory,newobj])
						*/
					maxErr = 1; //maxIter is 200
					//std::cout << "selectInit " << selectInit << " nIter " << nIter << std::endl;
					if(nIter >= maxIter){
						maxErr = 0;
						/*
						#if isfield(options,'Converge') and options.Converge:
						if "Converge" in options and options["Converge"]:
							pass
						else:
						*/
						objhistory.push_back(0);
					}
				}
			}
		}
		//#elapse = cputime - tmp_T
		//elapse = time.time() - tmp_T
		if(tryNo == 1){
			U_final = U;
			V_final = V;
			nIter_final = nIter;
			//elapse_final = elapse
			objhistory_final = objhistory;
			bSuccess["nStepTrial"] = nStepTrial;
		}else{
			if (!objhistory.empty() && !objhistory_final.empty() && (objhistory.back() < objhistory_final.back())) {
			//if (not isinstance(objhistory, list) and objhistory < objhistory_final) or \
			//(isinstance(objhistory,list) and objhistory[-1] < objhistory_final[-1]):
			//#if objhistory(end()) < objhistory_final(end()):
				U_final = U;
				V_final = V;
				nIter_final = nIter;
				objhistory_final = objhistory;
				bSuccess["nStepTrial"] = nStepTrial;
				/*
				if selectInit:
					elapse_final = elapse
				else:
					elapse_final = elapse_final + elapse
				*/
			}
		}
		if(selectInit){
			if(tryNo < nRepeat){
				U = Eigen::MatrixXd::NullaryExpr(mFea, k, [&](){return dis(gen);}).cwiseAbs();
				V = Eigen::MatrixXd::NullaryExpr(nSmp, k, [&](){return dis(gen);}).cwiseAbs();
				//U = Eigen::MatrixXd::Random(mFea, k).cwiseAbs(); // Random initialization
				//V = Eigen::MatrixXd::Random(nSmp, k).cwiseAbs(); // Random initialization
				NormalizePerView(U, V);
			
				//U = np.abs(np.random.rand(mFea,k))
				//V = np.abs(np.random.rand(nSmp,k))
				//U,V = Normalize(U,V)
				//print("Entering PerViewNMF, selectInit is true, tryNo < nRepeat")
				std::cout << "Entering PerViewNMF, selectInit is true, tryNo < nRepeat\n";

			}else{
				tryNo = tryNo - 1;
				minIter = 0;
				selectInit = false;
				U = U_final;
				V = V_final;
				objhistory = objhistory_final;
				meanFit = objhistory_final.back() * 10;
			}
		}
	}
	
	nIter_final = nIter_final + minIterOrig;
	NormalizePerView(U_final, V_final);

	res_nIter = nIter_final;
	res_U = U_final;
	res_V = V_final;
	res_objhistory = objhistory;
	//U_final,V_final = Normalize(U_final,V_final)
	//return U_final,V_final,nIter_final,elapse_final,bSuccess,objhistory_final
}	


int main(int argc, char* argv[]) {
	if(argc!=7){
		 std::cerr << "Usage: " << argv[0] << " <string> <nRepeat:integer> <minIterOrig:integer> <maxIter:integer> <outfile_string> <alpha:view weight>; if you want default values for param 2-4 and 6, please use -1. \n";
        return 1;
    }



    // Parse each argument
    std::string str_arg = argv[1]; // First argument as string
    int nRepeat = std::stoi(argv[2]); // Second argument as integer
    int minIterOrig = std::stoi(argv[3]); // Third argument as integer
    int maxIter = std::stoi(argv[4]); // Fourth argument as integer
	std::string output_filename = argv[5];
	int alphaView = std::stoi(argv[6]);

	if(nRepeat==-1){
		std::cerr << "Using default value for nRepeat\n";
		nRepeat = 30;
	}
	if(minIterOrig==-1){
		std::cerr << "Using default value for minIterOrig\n";
		minIterOrig = 50;
	}
	if(maxIter==-1){
		std::cerr << "Using default value for maxIter\n";
		maxIter = 200;
	}
	if(alphaView==-1){
		std::cerr << "Using default value for alphaWeight\n";
		alphaView = 1;
	}


    try {
        //std::string filename = "/project/sATAC/C1_peak_norm_expr.mat"; // Specify your .mat file name here
		std::string filename = str_arg;
        std::string varName = "expr"; // Specify your variable name here
        Eigen::MatrixXd matrix = readMatFile(filename, varName).transpose();
        std::cout << "Matrix read from .mat file:" << std::endl;
		std::cout << matrix.rows() << " " << matrix.cols() << std::endl;

		Eigen::MatrixXd morpho = readMatFile(filename, "morpho");
		std::cout << morpho.rows() << " " << morpho.cols() << std::endl;

		morpho = -1.0 * morpho;
		double Min = std::abs(morpho.minCoeff());
		morpho.array() += Min;

		std::vector<Eigen::MatrixXd> data(2);
		data[0] = matrix.transpose();
		data[1] = morpho.transpose();
		for(int i=0; i<2; i++){
			data[i] = data[i].array() / data[i].sum(); 
		}
	

		int num_factor = 20;	
		//int nRepeat = 30;
		//int minIterOrig = 50;
		//int maxIter = 200;

		std::map<std::string, int> bSuccess;

		Eigen::MatrixXd U_;
		Eigen::MatrixXd V_;
		std::vector<Eigen::MatrixXd> U(2);
		std::vector<Eigen::MatrixXd> V(2);
				
		int nIter = 0;
		std::vector<double> objhistory;

		std::cout << "init 0..."<<std::endl;
		NMF(data[0], num_factor, bSuccess, U_, V_, U[0], V[0], nIter, objhistory, nRepeat, minIterOrig);

		std::cout << "init 1..."<<std::endl;
		NMF(data[1], num_factor, bSuccess, U_, V[0], U[1], V[1], nIter, objhistory, nRepeat, minIterOrig);

		std::cout << "init 2..."<<std::endl;
		NMF(data[0], num_factor, bSuccess, U_, V[1], U[0], V[0], nIter, objhistory, nRepeat, minIterOrig);

		std::cout << "init 3..."<<std::endl;
		NMF(data[1], num_factor, bSuccess, U_, V[0], U[1], V[1], nIter, objhistory, nRepeat, minIterOrig);

		std::cout << "init 4..."<<std::endl;
		NMF(data[0], num_factor, bSuccess, U_, V[1], U[0], V[0], nIter, objhistory, nRepeat, minIterOrig);

		std::cout << "init 5..."<<std::endl;
		NMF(data[1], num_factor, bSuccess, U_, V[0], U[1], V[1], nIter, objhistory, nRepeat, minIterOrig);

		double oldL = 100;
		int j = 0;
		std::vector<double> log;

		//int Rounds = 30;
		int Rounds = nRepeat;
		Eigen::MatrixXd centroidV;
		std::vector<Eigen::MatrixXd> oldU(2);
		std::vector<Eigen::MatrixXd> oldV(2);

		std::vector<double> alpha(2);
		alpha[0] = 0.01;
		alpha[1] = 0.01 * alphaView;
		double alphaSum = 0;
		for(int i=0; i<alpha.size(); i++){
			alphaSum+=alpha[i];
		}

		Eigen::MatrixXd tmp1, tmp2;
		while(j < Rounds){
			j++;
			if(j==1){
				centroidV = V[0];
			}else{
				centroidV = alpha[0] * V[0].array();
				for(int i=1; i<2; i++){
					centroidV = centroidV.array() + alpha[i] * V[i].array();
				}
				centroidV = centroidV.array() / alphaSum;
			}
			double logL = 0;
			for(int i=0; i<2; i++){
				tmp1 = data[i].array() - (U[i] * V[i].transpose()).array();
				tmp2 = V[i].array() - centroidV.array();
				logL = logL + tmp1.squaredNorm() + alpha[i]*tmp2.squaredNorm();
			}
			log.push_back(logL);
			std::cout << logL << std::endl;
			//std::cout << "oldL " << oldL << " logL " << logL << std::endl;
			if(oldL < logL){
				U = oldU;
				V = oldV;
				logL = oldL;
				j--;
				std::cout << "restart this iteration\n";
			}
			oldU = U;
			oldV = V;
			oldL = logL;

			int res_nIter;
			std::vector<double> res_objhistory;
			for(int i=0; i<2; i++){
/*void PerViewNMF(const Eigen::MatrixXd& X, int k, Eigen::MatrixXd& Vo, 
Eigen::MatrixXd& U, Eigen::MatrixXd& V, 
Eigen::MatrixXd& res_U, Eigen::MatrixXd& res_V, 
int& res_nIter, std::vector<float>& res_objhistory){
*/				
				PerViewNMF(data[i], num_factor, centroidV, U[i], V[i], U[i], V[i], res_nIter, res_objhistory, nRepeat, minIterOrig, maxIter, alpha[i]);
			}
		}

//U[1],V[1],nIter,elapse,bSuccess,objhistory = NMF(X=data[1], k=num_factor, options=options, bSuccess=bSuccess, U_=U_, V_=V[0])
//U[0],V[0],nIter,elapse,bSuccess,objhistory = NMF(X=data[0], k=num_factor, options=options, bSuccess=bSuccess, U_=U_, V_=V[1])
		
		//previous good======================
		//saveMatrixVector(output_filename + "_U_final.bin", U);
		//saveMatrixVector(output_filename + "_V_final.bin", V);
		//saveMatrix(output_filename + "_V_centroid.bin", centroidV);

		saveMatrixVectorAsText(output_filename + "_U_final.txt", U);
		saveMatrixVectorAsText(output_filename + "_V_final.txt", V);
		saveMatrixAsText(output_filename + "_V_centroid.txt", centroidV);

	/*
std::map<std::string, int>& bSuccess,
Eigen::MatrixXd& U_, Eigen::MatrixXd& V_, 
Eigen::MatrixXd& res_U, Eigen::MatrixXd& res_V, 
int& res_nIter, std::vector<double>& res_objhistory){
		*/	
		//-1.0*mat["morpho"]
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    return 0;
}
