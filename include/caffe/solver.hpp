#ifndef CAFFE_OPTIMIZATION_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_SOLVER_HPP_

#include <string>
#include <vector>

#include "caffe/net.hpp"

namespace caffe {

/**
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ComputeUpdateValue to compute a parameter update
 * given the current state of the Net parameters.
 */
template <typename Dtype>
class Solver {
 public:
  explicit Solver(const SolverParameter& param);
  explicit Solver(const string& param_file);
  void Init(const SolverParameter& param);
  void InitTrainNet();
  void InitTestNets();
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  void Step(int iters);
  // The Restore function implements how one should restore the solver to a
  // previously snapshotted state. You should implement the RestoreSolverState()
  // function that restores the state from a SolverState protocol buffer.
  void Restore(const char* resume_file);
  virtual ~Solver() {}
  inline shared_ptr<Net<Dtype> > net() { return net_; }
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
    return test_nets_;
  }
  int iter() { return iter_; }
  int current_step() { return current_step_; }
  void setIter(int iter) { iter_ = iter; }
  void setCurrentStep(int step) { current_step_ = step; }

 public:
  // Get the update value for the current iteration.
  virtual void ComputeUpdateValue() = 0;
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  void Snapshot();
  // The test routine
  void TestAll();
  void Test(const int test_net_id = 0);
  virtual void SnapshotSolverState(SolverState* state) = 0;
  virtual void RestoreSolverState(const SolverState& state) = 0;
  void DisplayOutputBlobs(const int net_id);

 protected:
  SolverParameter param_;
  int iter_;
  int current_step_;
  shared_ptr<Net<Dtype> > net_;
  vector<shared_ptr<Net<Dtype> > > test_nets_;

  // smooth loss intermediate data
  vector<Dtype> losses;
  Dtype smoothed_loss;
  int stepped;  // times OneStep() invoked
  inline bool needDisplay() {
    return param_.display() && iter_ % param_.display() == 0;
  }

 public:
  void OneStep();
  void testPhase();
  void forwardBackwardPhase();
  void displayPhase();
  void snapshotPhase();
  void stepEnd();
  const SolverParameter& param() {return this->param_;}
  DISABLE_COPY_AND_ASSIGN(Solver);
};


/**
 * @brief Optimizes the parameters of a Net using
 *        stochastic gradient descent (SGD) with momentum.
 */
template <typename Dtype>
class SGDSolver : public Solver<Dtype> {
 public:
  explicit SGDSolver(const SolverParameter& param)
      : Solver<Dtype>(param) { PreSolve(); }
  explicit SGDSolver(const string& param_file)
      : Solver<Dtype>(param_file) { PreSolve(); }

  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }

 protected:
  void PreSolve();
  Dtype GetLearningRate();
  virtual void ComputeUpdateValue();
  virtual void ClipGradients();
  virtual void SnapshotSolverState(SolverState * state);
  virtual void RestoreSolverState(const SolverState& state);
  // history maintains the historical momentum data.
  // update maintains update related data and is not needed in snapshots.
  // temp maintains other information that might be needed in computation
  //   of gradients/updates and is not needed in snapshots
  vector<shared_ptr<Blob<Dtype> > > history_, update_, temp_;

  DISABLE_COPY_AND_ASSIGN(SGDSolver);
};

template <typename Dtype>
class NesterovSolver : public SGDSolver<Dtype> {
 public:
  explicit NesterovSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) {}
  explicit NesterovSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) {}

 protected:
  virtual void ComputeUpdateValue();

  DISABLE_COPY_AND_ASSIGN(NesterovSolver);
};

template <typename Dtype>
class AdaGradSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaGradSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { constructor_sanity_check(); }
  explicit AdaGradSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { constructor_sanity_check(); }

 protected:
  virtual void ComputeUpdateValue();
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with AdaGrad.";
  }

  DISABLE_COPY_AND_ASSIGN(AdaGradSolver);
};

template <typename Dtype>
class AdaDeltaSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaDeltaSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) {
    constructor_sanity_check();
    InitHistory();
  }
  explicit AdaDeltaSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) {
    constructor_sanity_check();
    InitHistory();
  }

 protected:
  void InitHistory();
  virtual void ComputeUpdateValue();
  virtual void SnapshotSolverState(SolverState * state);
  virtual void RestoreSolverState(const SolverState& state);

  void constructor_sanity_check() {
  }

  // history_ for momentum
  // rmsx for rmsx
  // rmsg for rmsg

  vector<shared_ptr<Blob<Dtype> > > rmsx_, rmsg_;

  DISABLE_COPY_AND_ASSIGN(AdaDeltaSolver);
};
template <typename Dtype>
class AdaMomentumSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaMomentumSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) {
    constructor_sanity_check();
    InitHistory();
  }
  explicit AdaMomentumSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) {
    constructor_sanity_check();
    InitHistory();
  }

 protected:
  void InitHistory();
  virtual void ComputeUpdateValue();
  virtual void SnapshotSolverState(SolverState * state);
  virtual void RestoreSolverState(const SolverState& state);

  void constructor_sanity_check() {
  }

  // history_ for momentum
  // rmsg for rmsg

  vector<shared_ptr<Blob<Dtype> > > rmsg_;

  DISABLE_COPY_AND_ASSIGN(AdaMomentumSolver);
};

template <typename Dtype>
Solver<Dtype>* GetSolver(const SolverParameter& param) {
  SolverParameter_SolverType type = param.solver_type();

  switch (type) {
  case SolverParameter_SolverType_SGD:
      return new SGDSolver<Dtype>(param);
  case SolverParameter_SolverType_NESTEROV:
      return new NesterovSolver<Dtype>(param);
  case SolverParameter_SolverType_ADAGRAD:
      return new AdaGradSolver<Dtype>(param);
  case SolverParameter_SolverType_ADADELTA:
      return new AdaDeltaSolver<Dtype>(param);
  case SolverParameter_SolverType_ADAMON:
      return new AdaMomentumSolver<Dtype>(param);
  default:
      LOG(FATAL) << "Unknown SolverType: " << type;
  }
  return (Solver<Dtype>*) NULL;
}

}  // namespace caffe

#endif  // CAFFE_OPTIMIZATION_SOLVER_HPP_
