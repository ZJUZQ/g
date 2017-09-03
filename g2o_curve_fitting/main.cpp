#include <iostream>
#include <cmath>
#include <ctime>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

using namespace std;

/**	Question: y = exp(a*x^2 + b*x +c) + w

	w is gauss noise;
	<x, y> is measurement data;
	a, b, c is the parameters to estimate
*/

/** 作为g2o的用户，我们要做的事主要包含一下步骤：
	1). 定义节点(优化变量)和边(误差)的类型;
	2). 构建图;
	3). 选择优化算法;
	4). 调用g2o进行优化，返回结果; */

 

/** 曲线模型的顶点
	模板参数：优化变量维度, 和数据类型 */

class CurveFittingVertex: public g2o::BaseVertex<3, Eigen::Vector3d>{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW // https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
	
	// sets the node to the origin, 重置
	virtual void setToOriginImpl(){
		_estimate << 0, 0, 0;
	}

	// update the position of the node, 更新
	virtual void oplusImpl(const double* update){
		_estimate += Eigen::Vector3d(update);
	}

	// 存盘和读盘：留空
	virtual bool read( istream& in ) {}
	virtual bool write(  ostream& out ) const {}
};

/** 误差模型 
	模板参数：观测值维度，类型，连接顶点类型 */

class CurveFittingEdge: public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	CurveFittingEdge(double x): BaseUnaryEdge(), _x(x) {}

	// 计算曲线模型误差
	void computeError(){
		const CurveFittingVertex* v = static_cast<const CurveFittingVertex*> (_vertices[0]);
		const Eigen::Vector3d abc = v->estimate();
		_error(0, 0) = _measurement - std::exp( abc(0, 0)*_x*_x + abc(1, 0)*_x + abc(2, 0) );
	}

	virtual bool read( istream& in ) {}
    virtual bool write( ostream& out ) const {}

public:
	double _x; // x 值， y 值为 _measurement
};


int main(int argc, char** argv){

	double a=1.0, b=2.0, c=1.0;         // 真实参数值
    int N=100;                          // 数据点
    double w_sigma=1.0;                 // 噪声Sigma值
    cv::RNG rng;                        // OpenCV随机数产生器
    double abc[3] = {0,0,0};            // abc参数的估计值

    std::vector<double> x_data, y_data; // 数据

    cout << "generating data: " << endl;
    for(int i = 0; i < N; i++){
    	double x = i / 100.0;
    	x_data.push_back(x);
    	y_data.push_back(  std::exp(a*x*x + b*x + c) + rng.gaussian(w_sigma) );
    }

    // 构建图优化，先设定g2o

    /** struct BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> {
			static const int PoseDim = Eigen::Dynamic;
    		static const int LandmarkDim = Eigen::Dynamic;
    	} */
    // 每个误差项优化变量维度为3，误差值维度为1
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<3, 1> > Block;
    // 线性方程求解器
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    // 矩阵块求解器
    Block* solver_ptr = new Block(linearSolver);
    // 梯度下降方法，从GN, LM, DogLeg 中选
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
    // g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );

    g2o::SparseOptimizer optimizer; 	// 图模型
    optimizer.setAlgorithm(solver);		// 设置求解器
    optimizer.setVerbose(true);			// 打开调试输出

    // 往图中增加顶点
    CurveFittingVertex* v = new CurveFittingVertex();
    v->setEstimate( Eigen::Vector3d(0, 0, 0) );
    v->setId(0);
    optimizer.addVertex( v );

    // 往图中增加边
    for(int i = 0; i < N; i++){
    	CurveFittingEdge* edge = new CurveFittingEdge( x_data[i] );
    	edge->setId(i);
    	/**	 set the ith vertex on the hyper-edge to the pointer supplied
    		 超边(Hyper Edge), 又称为一元边(Unary Edge)，即只连接一个顶点
            
          	void setVertex(size_t i, Vertex* v) { 
          		assert(i < _vertices.size() && "index out of bounds"); 
          		_vertices[i]=v;
          	}
    	*/
    	edge->setVertex(0, v); 				// 设置连接的顶点
    	edge->setMeasurement( y_data[i] ); 	// 观测数值
    	edge->setInformation( Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma) ); // 信息矩阵：协方差矩阵之逆
    	optimizer.addEdge( edge );
    }

    // 执行优化
    cout << "start optimization" << endl;
    clock_t t1 = clock();
    optimizer.initializeOptimization();
    /**
    	int optimize(int iterations, bool online = false);
    */
    optimizer.optimize(100);
    clock_t t2 = clock();
    cout << "slove time cost = " << (t2 - t1) * 1000 / CLOCKS_PER_SEC << " ms\n";

    // 输出优化值
    Eigen::Vector3d abc_estimated = v->estimate(); // return the current estimate of the vertex
    cout << "estimeated model: " << abc_estimated.transpose() << endl;
    return 0; 
}


