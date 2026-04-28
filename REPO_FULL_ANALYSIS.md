# Repository Deep Dive (Architecture + File-by-File Guide)

## 1. What This Repository Is

This repository combines the DeepXDE framework source with a custom local research project (`myproject`).

- Framework core: `deepxde/`
- Official docs: `docs/`
- Official runnable examples: `examples/`
- Custom engineering/research workflow: `myproject/`

## 2. Architecture Overview

### 2.1 Layered Structure

1. Backend abstraction: `deepxde/backend`
2. Problem modeling: `deepxde/geometry`, `deepxde/icbc`, `deepxde/data`
3. Network and optimization: `deepxde/nn`, `deepxde/optimizers`
4. Training runtime: `deepxde/model.py`, `deepxde/callbacks.py`
5. Utilities and display: `deepxde/utils`, `deepxde/display.py`
6. ZCS extension for operator learning: `deepxde/zcs`

### 2.2 Main Training Flow (Framework)

`geometry -> icbc -> data -> nn -> Model.compile -> Model.train`

### 2.3 Main Training Flow (myproject)

`pinn_training_data.mat -> physics baseline force -> Cd(Re) fitting -> residual MLP correction -> run artifacts`

## 3. Engineering Notes

- Strong modular boundaries in the framework core.
- Good multi-backend portability design.
- `myproject/runs_force` has reproducible run snapshots.
- Mixed source+artifact repository layout may add long-term VCS noise.

## 4. File-by-File Notes (406 tracked files)

### <root>

- `.codacy.yml`: Codacy quality scan configuration (duplication checks and path exclusions).
- `.gitignore`: Git ignore rules for build artifacts, caches, and IDE files.
- `.prospector.yaml`: Prospector/Pylint/Pep257 static analysis rules.
- `.readthedocs.yaml`: ReadTheDocs build configuration and docs install steps.
- `CITATION.cff`: Citation metadata (authors, DOI, repository URL).
- `LICENSE`: LGPL-2.1 license text.
- `README.md`: Repository overview, features, install instructions, and references.
- `check_environment.py`: Local environment diagnostic script (Python, env vars, import path).
- `diagnose_deepxde.py`: DeepXDE diagnostic script for backend/import/basic object checks.
- `loss.dat`: Sample loss output data file.
- `output.txt`: Sample console/output log file.
- `pyproject.toml`: Packaging metadata and build backend config.
- `requirements.txt`: Base runtime dependencies.
- `run_euler_beam.bat`: Windows helper script to set backend and run myproject Euler beam case.
- `test.dat`: Sample test output data file.
- `test_deepxde.py`: Minimal smoke test to verify DeepXDE import and basic API usage.
- `test_output.txt`: Test output placeholder file.
- `test_output2.txt`: Test output placeholder file.
- `train.dat`: Sample training output data file.
- `version.txt`: Version placeholder file (currently empty).

### .github

- `.github/ISSUE_TEMPLATE/config.yml`: Issue template/discussion redirect configuration.
- `.github/workflows/build.yml`: CI workflow: multi-OS, multi-Python, multi-backend import smoke checks.
- `.github/workflows/release.yml`: Release workflow: build wheel/sdist and publish to PyPI.

### .specstory

- `.specstory/.what-is-this.md`: SpecStory folder description and usage notes.
- `.specstory/history/2025-07-04_09-40Z-切换文件夹时光标历史记录消失.md`: SpecStory auto-saved AI conversation history file.
- `.specstory/history/2025-07-04_16-00Z-解释-euler-beam-py-代码及其控制方程.md`: SpecStory auto-saved AI conversation history file.
- `.specstory/history/2025-07-09_08-04Z-使用-mat文件训练模型.md`: SpecStory auto-saved AI conversation history file.

### deepxde

- `deepxde/__init__.py`: Top-level DeepXDE package entry; exports public API objects and modules.
- `deepxde/backend/__init__.py`: Backend dispatcher: resolves DDE_BACKEND and loads framework adapter. (funcs: is_enabled, _gen_missing_api, backend_message, load_backend ...)
- `deepxde/backend/backend.py`: Unified backend interface contract (tensor ops, math ops, linalg, regularization). (funcs: data_type_dict, is_gpu_available, is_tensor, shape ...)
- `deepxde/backend/jax/__init__.py`: Backend subpackage entry.
- `deepxde/backend/jax/tensor.py`: jax backend adapter implementation for the unified backend API. (classes: Variable; funcs: data_type_dict, is_tensor, shape, ndim ...)
- `deepxde/backend/paddle/__init__.py`: Backend subpackage entry.
- `deepxde/backend/paddle/tensor.py`: paddle backend adapter implementation for the unified backend API. (funcs: data_type_dict, is_gpu_available, is_tensor, shape ...)
- `deepxde/backend/pytorch/__init__.py`: Backend subpackage entry.
- `deepxde/backend/pytorch/tensor.py`: pytorch backend adapter implementation for the unified backend API. (funcs: data_type_dict, is_gpu_available, is_tensor, shape ...)
- `deepxde/backend/set_default_backend.py`: CLI/helper to persist default backend in ~/.deepxde/config.json. (funcs: set_default_backend)
- `deepxde/backend/tensorflow/__init__.py`: Backend subpackage entry.
- `deepxde/backend/tensorflow/tensor.py`: tensorflow backend adapter implementation for the unified backend API. (funcs: data_type_dict, is_gpu_available, is_tensor, shape ...)
- `deepxde/backend/tensorflow_compat_v1/__init__.py`: Backend subpackage entry.
- `deepxde/backend/tensorflow_compat_v1/tensor.py`: tensorflow_compat_v1 backend adapter implementation for the unified backend API. (funcs: data_type_dict, is_gpu_available, is_tensor, shape ...)
- `deepxde/backend/utils.py`: Backend probing and install helpers (including Paddle install path). (funcs: import_tensorflow_compat_v1, import_tensorflow, import_pytorch, import_jax ...)
- `deepxde/callbacks.py`: Training callback system (checkpoint, early stop, resampling, uncertainty, diagnostics). (classes: Callback, CallbackList, ModelCheckpoint ...)
- `deepxde/config.py`: Global runtime configuration (precision, seed, XLA, parallel/Horovod). (funcs: default_float, set_default_float, set_random_seed, enable_xla_jit ...)
- `deepxde/data/__init__.py`: Data package exports (PDE/FPDE/IDE/operator/multifidelity/samplers).
- `deepxde/data/constraint.py`: Constraint-data object for constrained training setups. (classes: Constraint)
- `deepxde/data/data.py`: Abstract Data base class and tuple data container. (classes: Data, Tuple)
- `deepxde/data/dataset.py`: Supervised dataset wrapper with split/transform support. (classes: DataSet)
- `deepxde/data/fpde.py`: Fractional PDE (fPINN) data/discretization and integral matrix construction. (classes: Scheme, FPDE, TimeFPDE ...)
- `deepxde/data/func_constraint.py`: Function-constraint data object. (classes: FuncConstraint)
- `deepxde/data/function.py`: Function approximation data object. (classes: Function)
- `deepxde/data/function_spaces.py`: Function-space generators (PowerSeries/Chebyshev/GRF/etc.). (classes: FunctionSpace, PowerSeries, Chebyshev ...; funcs: wasserstein2, eig)
- `deepxde/data/helper.py`: Small data helper functions (constant one/zero function generators). (funcs: zero_function, one_function)
- `deepxde/data/ide.py`: Integro-differential equation data object. (classes: IDE)
- `deepxde/data/mf.py`: Multifidelity data objects. (classes: MfFunc, MfDataSet)
- `deepxde/data/pde.py`: Core PDE/TimePDE data pipeline (sampling, BC/PDE split, losses). (classes: PDE, TimePDE)
- `deepxde/data/pde_operator.py`: PI-DeepONet operator data pipelines. (classes: PDEOperator, PDEOperatorCartesianProd)
- `deepxde/data/quadruple.py`: Quadruple dataset structures (and Cartesian product variant). (classes: Quadruple, QuadrupleCartesianProd)
- `deepxde/data/sampler.py`: Batch sampler utility. (classes: BatchSampler)
- `deepxde/data/triple.py`: Triple dataset structures (and Cartesian product variant). (classes: Triple, TripleCartesianProd)
- `deepxde/display.py`: Training progress and best-result display formatting. (classes: TrainingDisplay)
- `deepxde/geometry/__init__.py`: Geometry package exports (1D/2D/3D/ND/time/CSG).
- `deepxde/geometry/csg.py`: Constructive Solid Geometry operators (union/difference/intersection). (classes: CSGUnion, CSGDifference, CSGIntersection)
- `deepxde/geometry/geometry.py`: Abstract geometry contract (inside/boundary/sampling APIs). (classes: Geometry)
- `deepxde/geometry/geometry_1d.py`: 1D geometry implementation (Interval). (classes: Interval)
- `deepxde/geometry/geometry_2d.py`: 2D geometry implementations (Disk, Ellipse, Rectangle, Triangle, Polygon, StarShaped). (classes: Disk, Ellipse, Rectangle ...; funcs: polygon_signed_area, clockwise_rotation_90, is_left, is_rectangle ...)
- `deepxde/geometry/geometry_3d.py`: 3D geometry implementations (Cuboid, Sphere). (classes: Cuboid, Sphere)
- `deepxde/geometry/geometry_nd.py`: ND geometry implementations (Hypercube, Hypersphere). (classes: Hypercube, Hypersphere)
- `deepxde/geometry/pointcloud.py`: Point-cloud geometry implementation. (classes: PointCloud)
- `deepxde/geometry/sampler.py`: Geometry sampling strategies (pseudo/random/quasi-random). (funcs: sample, pseudorandom, quasirandom)
- `deepxde/geometry/timedomain.py`: Time-domain and geometry-time composition. (classes: TimeDomain, GeometryXTime)
- `deepxde/gradients/__init__.py`: Gradient API export (jacobian/hessian).
- `deepxde/gradients/gradients.py`: Gradient frontend that switches forward/reverse autodiff mode. (funcs: jacobian, hessian, clear)
- `deepxde/gradients/gradients_forward.py`: Forward-mode autodiff implementation. (classes: JacobianForward; funcs: jacobian, hessian, clear)
- `deepxde/gradients/gradients_reverse.py`: Reverse-mode autodiff implementation. (classes: JacobianReverse, Hessian, Hessians; funcs: jacobian, hessian, clear)
- `deepxde/gradients/jacobian.py`: Lazy Jacobian/Hessian cache and helper abstractions. (classes: Jacobian, Jacobians)
- `deepxde/icbc/__init__.py`: IC/BC package export entry.
- `deepxde/icbc/boundary_conditions.py`: Boundary condition implementations (Dirichlet/Neumann/Robin/Periodic/Operator/PointSet/Interface2D). (classes: BC, DirichletBC, NeumannBC ...; funcs: npfunc_range_autocache)
- `deepxde/icbc/initial_conditions.py`: Initial condition implementation. (classes: IC)
- `deepxde/losses.py`: Loss function registry and resolver. (funcs: mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, mean_l2_relative_error ...)
- `deepxde/metrics.py`: Metric function registry and resolver. (funcs: accuracy, l2_relative_error, nanl2_relative_error, mean_l2_relative_error ...)
- `deepxde/model.py`: Core training runtime: compile/train/predict/save/restore with backend branching. (classes: Model, TrainState, LossHistory)
- `deepxde/nn/__init__.py`: NN dispatcher that loads backend-specific network implementations. (funcs: _load_backend)
- `deepxde/nn/activations.py`: Activation registry and L-LAAF helper. (funcs: linear, layer_wise_locally_adaptive, get)
- `deepxde/nn/deeponet_strategy.py`: DeepONet multi-output split/merge strategies. (classes: DeepONetStrategy, SingleOutputStrategy, IndependentStrategy ...)
- `deepxde/nn/initializers.py`: Initializer registries across backends. (classes: VarianceScalingStacked; funcs: _compute_fans_stacked, initializer_dict_tf, initializer_dict_torch, initializer_dict_jax ...)
- `deepxde/nn/jax/__init__.py`: jax NN package export entry.
- `deepxde/nn/jax/fnn.py`: jax network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: FNN, PFNN)
- `deepxde/nn/jax/nn.py`: jax network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: NN)
- `deepxde/nn/paddle/__init__.py`: paddle NN package export entry.
- `deepxde/nn/paddle/deeponet.py`: paddle network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: DeepONet, DeepONetCartesianProd)
- `deepxde/nn/paddle/fnn.py`: paddle network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: FNN, PFNN)
- `deepxde/nn/paddle/mfnn.py`: paddle network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: MfNN)
- `deepxde/nn/paddle/msffn.py`: paddle network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: MsFFN, STMsFFN)
- `deepxde/nn/paddle/nn.py`: paddle network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: NN)
- `deepxde/nn/pytorch/__init__.py`: pytorch NN package export entry.
- `deepxde/nn/pytorch/deeponet.py`: pytorch network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: DeepONet, DeepONetCartesianProd, PODDeepONet)
- `deepxde/nn/pytorch/fnn.py`: pytorch network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: FNN, PFNN)
- `deepxde/nn/pytorch/mionet.py`: pytorch network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: MIONetCartesianProd, PODMIONet)
- `deepxde/nn/pytorch/nn.py`: pytorch network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: NN)
- `deepxde/nn/regularizers.py`: Regularizer factory. (funcs: get)
- `deepxde/nn/tensorflow/__init__.py`: tensorflow NN package export entry.
- `deepxde/nn/tensorflow/deeponet.py`: tensorflow network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: DeepONet, DeepONetCartesianProd, PODDeepONet)
- `deepxde/nn/tensorflow/fnn.py`: tensorflow network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: FNN, PFNN)
- `deepxde/nn/tensorflow/nn.py`: tensorflow network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: NN)
- `deepxde/nn/tensorflow_compat_v1/__init__.py`: tensorflow_compat_v1 NN package export entry.
- `deepxde/nn/tensorflow_compat_v1/deeponet.py`: tensorflow_compat_v1 network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: DeepONetStrategy, SingleOutputStrategy, IndependentStrategy ...)
- `deepxde/nn/tensorflow_compat_v1/fnn.py`: tensorflow_compat_v1 network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: FNN, PFNN)
- `deepxde/nn/tensorflow_compat_v1/mfnn.py`: tensorflow_compat_v1 network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: MfNN)
- `deepxde/nn/tensorflow_compat_v1/mionet.py`: tensorflow_compat_v1 network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: MIONet, MIONetCartesianProd)
- `deepxde/nn/tensorflow_compat_v1/msffn.py`: tensorflow_compat_v1 network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: MsFFN, STMsFFN)
- `deepxde/nn/tensorflow_compat_v1/nn.py`: tensorflow_compat_v1 network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: NN)
- `deepxde/nn/tensorflow_compat_v1/resnet.py`: tensorflow_compat_v1 network implementation module (FNN/DeepONet/MIONet/MfNN/MsFFN/base NN by file). (classes: ResNet)
- `deepxde/optimizers/__init__.py`: Optimizer dispatcher that loads backend-specific optimizer logic. (funcs: _load_backend)
- `deepxde/optimizers/config.py`: Global optimizer option registry (L-BFGS/NNCG/Horovod settings). (funcs: set_LBFGS_options, set_NNCG_options, set_hvd_opt_options)
- `deepxde/optimizers/jax/__init__.py`: jax optimizer package export entry.
- `deepxde/optimizers/jax/optimizers.py`: jax optimizer implementation module (optimizers/schedulers/external wrappers). (funcs: is_external_optimizer, get, _get_learningrate)
- `deepxde/optimizers/paddle/__init__.py`: paddle optimizer package export entry.
- `deepxde/optimizers/paddle/optimizers.py`: paddle optimizer implementation module (optimizers/schedulers/external wrappers). (funcs: _get_lr_scheduler, is_external_optimizer, get)
- `deepxde/optimizers/pytorch/__init__.py`: pytorch optimizer package export entry.
- `deepxde/optimizers/pytorch/nncg.py`: pytorch optimizer implementation module (optimizers/schedulers/external wrappers). (classes: NNCG; funcs: _armijo, _apply_nys_precond_inv, _nystrom_pcg)
- `deepxde/optimizers/pytorch/optimizers.py`: pytorch optimizer implementation module (optimizers/schedulers/external wrappers). (funcs: is_external_optimizer, get, _get_learningrate_scheduler)
- `deepxde/optimizers/tensorflow/__init__.py`: tensorflow optimizer package export entry.
- `deepxde/optimizers/tensorflow/optimizers.py`: tensorflow optimizer implementation module (optimizers/schedulers/external wrappers). (funcs: is_external_optimizer, get, _get_learningrate)
- `deepxde/optimizers/tensorflow/tfp_optimizer.py`: tensorflow optimizer implementation module (optimizers/schedulers/external wrappers). (classes: LossAndFlatGradient; funcs: lbfgs_minimize)
- `deepxde/optimizers/tensorflow_compat_v1/__init__.py`: tensorflow_compat_v1 optimizer package export entry.
- `deepxde/optimizers/tensorflow_compat_v1/optimizers.py`: tensorflow_compat_v1 optimizer implementation module (optimizers/schedulers/external wrappers). (funcs: is_external_optimizer, get, _get_learningrate)
- `deepxde/optimizers/tensorflow_compat_v1/scipy_optimizer.py`: tensorflow_compat_v1 optimizer implementation module (optimizers/schedulers/external wrappers). (classes: ExternalOptimizerInterface, ScipyOptimizerInterface; funcs: _accumulate, _get_shape_tuple, _prod, _compute_gradients)
- `deepxde/optimizers/tensorflow_compat_v1/tfp_optimizer.py`: tensorflow_compat_v1 optimizer implementation module (optimizers/schedulers/external wrappers). (classes: LossAndFlatGradient; funcs: plot_helper)
- `deepxde/real.py`: Float precision mapping helper. (classes: Real)
- `deepxde/utils/__init__.py`: Utility aggregate entry (internal + external + backend-specific helpers). (funcs: _load_backend)
- `deepxde/utils/array_ops_compat.py`: Cross-backend array compatibility helpers. (funcs: istensorlist, convert_to_array, hstack, roll ...)
- `deepxde/utils/external.py`: External-facing utilities (save/plot/standardize/data export/point sets). (classes: PointSet; funcs: apply, standardize, uniformly_continuous_delta, saveplot ...)
- `deepxde/utils/internal.py`: Internal shared helpers (decorators, checks, timing, MPI helpers). (funcs: timing, run_if_all_none, run_if_any_none, vectorize ...)
- `deepxde/utils/jax.py`: JAX-specific utility hooks.
- `deepxde/utils/paddle.py`: Paddle-specific utility hooks.
- `deepxde/utils/pytorch.py`: PyTorch-specific utility hooks. (classes: LLAAF)
- `deepxde/utils/tensorflow.py`: TensorFlow-specific utility hooks.
- `deepxde/utils/tensorflow_compat_v1.py`: TF1 compatibility utility hooks. (funcs: guarantee_initialized_variables)
- `deepxde/zcs/__init__.py`: ZCS (Zero Coordinate Shift) feature export entry.
- `deepxde/zcs/gradient.py`: ZCS gradient helper class implementation. (classes: LazyGrad)
- `deepxde/zcs/model.py`: Model extension for ZCS mode. (classes: Model)
- `deepxde/zcs/operator.py`: Operator extension for ZCS mode. (classes: PDEOperatorCartesianProd)

### docs

- `docs/Makefile`: Docs build scripts (including multi-backend tutorial build targets).
- `docs/conf.py`: Sphinx configuration (theme/extensions/version/build behavior).
- `docs/demos/function.rst`: Demo/tutorial page: function.
- `docs/demos/function/dataset.rst`: Demo/tutorial page: dataset.
- `docs/demos/function/func.rst`: Demo/tutorial page: func.
- `docs/demos/operator.rst`: Demo/tutorial page: operator.
- `docs/demos/operator/antiderivative_aligned.rst`: Demo/tutorial page: antiderivative_aligned.
- `docs/demos/operator/antiderivative_unaligned.rst`: Demo/tutorial page: antiderivative_unaligned.
- `docs/demos/operator/poisson.1d.pideeponet.rst`: Demo/tutorial page: poisson.1d.pideeponet.
- `docs/demos/operator/poisson_1d_pideeponet.png`: Demo/tutorial page: poisson_1d_pideeponet.
- `docs/demos/operator/stokes.png`: Demo/tutorial page: stokes.
- `docs/demos/operator/zcs.rst`: Demo/tutorial page: zcs.
- `docs/demos/pinn_forward.rst`: Demo/tutorial page: pinn_forward.
- `docs/demos/pinn_forward/Kovasznay.flow.rst`: Demo/tutorial page: Kovasznay.flow.
- `docs/demos/pinn_forward/allen.cahn.rst`: Demo/tutorial page: allen.cahn.
- `docs/demos/pinn_forward/burgers.rar.rst`: Demo/tutorial page: burgers.rar.
- `docs/demos/pinn_forward/burgers.rst`: Demo/tutorial page: burgers.
- `docs/demos/pinn_forward/diffusion.1d.exactBC.rst`: Demo/tutorial page: diffusion.1d.exactBC.
- `docs/demos/pinn_forward/diffusion.1d.resample.rst`: Demo/tutorial page: diffusion.1d.resample.
- `docs/demos/pinn_forward/diffusion.1d.rst`: Demo/tutorial page: diffusion.1d.
- `docs/demos/pinn_forward/diffusion.reaction.rst`: Demo/tutorial page: diffusion.reaction.
- `docs/demos/pinn_forward/elasticity.plate.rst`: Demo/tutorial page: elasticity.plate.
- `docs/demos/pinn_forward/eulerbeam.rst`: Demo/tutorial page: eulerbeam.
- `docs/demos/pinn_forward/heat.resample.rst`: Demo/tutorial page: heat.resample.
- `docs/demos/pinn_forward/heat.rst`: Demo/tutorial page: heat.
- `docs/demos/pinn_forward/helmholtz.2d.dirichlet.hpo.rst`: Demo/tutorial page: helmholtz.2d.dirichlet.hpo.
- `docs/demos/pinn_forward/helmholtz.2d.dirichlet.rst`: Demo/tutorial page: helmholtz.2d.dirichlet.
- `docs/demos/pinn_forward/helmholtz.2d.neumann.hole.rst`: Demo/tutorial page: helmholtz.2d.neumann.hole.
- `docs/demos/pinn_forward/helmholtz.2d.sound.hard.abc.rst`: Demo/tutorial page: helmholtz.2d.sound.hard.abc.
- `docs/demos/pinn_forward/klein.gordon.rst`: Demo/tutorial page: klein.gordon.
- `docs/demos/pinn_forward/laplace.disk.rst`: Demo/tutorial page: laplace.disk.
- `docs/demos/pinn_forward/lotka.volterra.rst`: Demo/tutorial page: lotka.volterra.
- `docs/demos/pinn_forward/ode.2nd.rst`: Demo/tutorial page: ode.2nd.
- `docs/demos/pinn_forward/ode.system.rst`: Demo/tutorial page: ode.system.
- `docs/demos/pinn_forward/poisson.1d.dirichlet.rst`: Demo/tutorial page: poisson.1d.dirichlet.
- `docs/demos/pinn_forward/poisson.1d.dirichletperiodic.rst`: Demo/tutorial page: poisson.1d.dirichletperiodic.
- `docs/demos/pinn_forward/poisson.1d.dirichletrobin.rst`: Demo/tutorial page: poisson.1d.dirichletrobin.
- `docs/demos/pinn_forward/poisson.1d.multiscaleFourier.rst`: Demo/tutorial page: poisson.1d.multiscaleFourier.
- `docs/demos/pinn_forward/poisson.1d.neumanndirichlet.rst`: Demo/tutorial page: poisson.1d.neumanndirichlet.
- `docs/demos/pinn_forward/poisson.1d.pointsetoperator.rst`: Demo/tutorial page: poisson.1d.pointsetoperator.
- `docs/demos/pinn_forward/poisson.Lshape.rst`: Demo/tutorial page: poisson.Lshape.
- `docs/demos/pinn_forward/poisson.dirichlet.1d.exactbc.rst`: Demo/tutorial page: poisson.dirichlet.1d.exactbc.
- `docs/demos/pinn_inverse.rst`: Demo/tutorial page: pinn_inverse.
- `docs/demos/pinn_inverse/diffusion.1d.inverse.rst`: Demo/tutorial page: diffusion.1d.inverse.
- `docs/demos/pinn_inverse/elliptic.inverse.field.rst`: Demo/tutorial page: elliptic.inverse.field.
- `docs/demos/pinn_inverse/lorenz.inverse.forced.rst`: Demo/tutorial page: lorenz.inverse.forced.
- `docs/demos/pinn_inverse/lorenz.inverse.rst`: Demo/tutorial page: lorenz.inverse.
- `docs/demos/pinn_inverse/reaction.inverse.rst`: Demo/tutorial page: reaction.inverse.
- `docs/images/backend.png`: Documentation image assets.
- `docs/images/dataparallel.png`: Documentation image assets.
- `docs/images/deeponet.png`: Documentation image assets.
- `docs/images/mfnn.png`: Documentation image assets.
- `docs/images/pinn.png`: Documentation image assets.
- `docs/images/scaling.png`: Documentation image assets.
- `docs/index.rst`: Docs homepage and table-of-contents root.
- `docs/make.bat`: Docs build scripts (including multi-backend tutorial build targets).
- `docs/modules/deepxde.backend.jax.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.backend.paddle.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.backend.pytorch.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.backend.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.backend.tensorflow.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.backend.tensorflow_compat_v1.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.data.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.geometry.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.gradients.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.icbc.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.nn.jax.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.nn.paddle.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.nn.pytorch.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.nn.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.nn.tensorflow.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.nn.tensorflow_compat_v1.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.optimizers.jax.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.optimizers.paddle.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.optimizers.pytorch.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.optimizers.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.optimizers.tensorflow.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.optimizers.tensorflow_compat_v1.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.rst`: API autodoc bridge pages for modules/packages.
- `docs/modules/deepxde.utils.rst`: API autodoc bridge pages for modules/packages.
- `docs/requirements.txt`: Documentation build dependencies.
- `docs/user/cite_deepxde.rst`: User-guide docs page (install/parallel/FAQ/research/team/citation topics).
- `docs/user/faq.rst`: User-guide docs page (install/parallel/FAQ/research/team/citation topics).
- `docs/user/installation.rst`: User-guide docs page (install/parallel/FAQ/research/team/citation topics).
- `docs/user/parallel.rst`: User-guide docs page (install/parallel/FAQ/research/team/citation topics).
- `docs/user/research.rst`: User-guide docs page (install/parallel/FAQ/research/team/citation topics).
- `docs/user/team.rst`: User-guide docs page (install/parallel/FAQ/research/team/citation topics).

### examples

- `examples/Makefile`: Batch example-test runner (converts demos to quick integration tests).
- `examples/dataset/Allen_Cahn.mat`: Example dataset asset file (mat/npz/dat).
- `examples/dataset/Burgers.npz`: Example dataset asset file (mat/npz/dat).
- `examples/dataset/Lorenz.npz`: Example dataset asset file (mat/npz/dat).
- `examples/dataset/Poisson_Lshape.npz`: Example dataset asset file (mat/npz/dat).
- `examples/dataset/cylinder_nektar_wake.mat`: Example dataset asset file (mat/npz/dat).
- `examples/dataset/dataset.test`: Example dataset asset file (mat/npz/dat).
- `examples/dataset/dataset.train`: Example dataset asset file (mat/npz/dat).
- `examples/dataset/mf_hi_test.dat`: Example dataset asset file (mat/npz/dat).
- `examples/dataset/mf_hi_train.dat`: Example dataset asset file (mat/npz/dat).
- `examples/dataset/mf_lo_train.dat`: Example dataset asset file (mat/npz/dat).
- `examples/dataset/reaction.npz`: Example dataset asset file (mat/npz/dat).
- `examples/dataset/stokes.npz`: Example dataset asset file (mat/npz/dat).
- `examples/function/dataset.py`: function approximation example script: dataset.
- `examples/function/func.py`: function approximation example script: func. (funcs: func)
- `examples/function/func_uncertainty.py`: function approximation example script: func_uncertainty. (funcs: func)
- `examples/function/mf_dataset.py`: function approximation example script: mf_dataset.
- `examples/function/mf_func.gpi`: Gnuplot script for example visualization: mf_func.
- `examples/function/mf_func.py`: function approximation example script: mf_func. (funcs: func_lo, func_hi)
- `examples/operator/ADR_solver.py`: operator learning (DeepONet/PI-DeepONet) example script: ADR_solver. (funcs: solve_ADR, main)
- `examples/operator/advection_aligned_pideeponet.py`: operator learning (DeepONet/PI-DeepONet) example script: advection_aligned_pideeponet. (funcs: pde, func_ic, periodic)
- `examples/operator/advection_aligned_pideeponet_2d.py`: operator learning (DeepONet/PI-DeepONet) example script: advection_aligned_pideeponet_2d. (funcs: pde, func_ic, boundary, periodic)
- `examples/operator/advection_unaligned_pideeponet.py`: operator learning (DeepONet/PI-DeepONet) example script: advection_unaligned_pideeponet. (funcs: pde, func_ic, periodic)
- `examples/operator/advection_unaligned_pideeponet_2d.py`: operator learning (DeepONet/PI-DeepONet) example script: advection_unaligned_pideeponet_2d. (funcs: pde, func_ic, boundary, periodic)
- `examples/operator/antiderivative_aligned.py`: operator learning (DeepONet/PI-DeepONet) example script: antiderivative_aligned.
- `examples/operator/antiderivative_aligned_pideeponet.py`: operator learning (DeepONet/PI-DeepONet) example script: antiderivative_aligned_pideeponet. (funcs: pde, zero_ic)
- `examples/operator/antiderivative_unaligned.py`: operator learning (DeepONet/PI-DeepONet) example script: antiderivative_unaligned.
- `examples/operator/antiderivative_unaligned_pideeponet.py`: operator learning (DeepONet/PI-DeepONet) example script: antiderivative_unaligned_pideeponet. (funcs: pde, zero_ic)
- `examples/operator/diff_rec_aligned_pideeponet.py`: operator learning (DeepONet/PI-DeepONet) example script: diff_rec_aligned_pideeponet. (funcs: pde)
- `examples/operator/diff_rec_aligned_zcs_pideeponet.py`: operator learning (DeepONet/PI-DeepONet) example script: diff_rec_aligned_zcs_pideeponet. (funcs: pde)
- `examples/operator/diff_rec_unaligned_pideeponet.py`: operator learning (DeepONet/PI-DeepONet) example script: diff_rec_unaligned_pideeponet. (funcs: pde)
- `examples/operator/poisson_1d_pideeponet.py`: operator learning (DeepONet/PI-DeepONet) example script: poisson_1d_pideeponet. (funcs: equation, u_boundary, boundary)
- `examples/operator/stokes_aligned_pideeponet.py`: operator learning (DeepONet/PI-DeepONet) example script: stokes_aligned_pideeponet. (funcs: pde, bc_slip_top_func, out_transform, plot_sol)
- `examples/operator/stokes_aligned_zcs_pideeponet.py`: operator learning (DeepONet/PI-DeepONet) example script: stokes_aligned_zcs_pideeponet. (funcs: pde, bc_slip_top_func, out_transform, plot_sol)
- `examples/pinn_forward/Allen_Cahn.py`: forward PINN example script: Allen_Cahn. (funcs: gen_testdata, pde, output_transform)
- `examples/pinn_forward/Beltrami_flow.py`: forward PINN example script: Beltrami_flow. (funcs: pde, u_func, v_func, w_func ...)
- `examples/pinn_forward/Burgers.py`: forward PINN example script: Burgers. (funcs: gen_testdata, pde)
- `examples/pinn_forward/Burgers_RAR.py`: forward PINN example script: Burgers_RAR. (funcs: gen_testdata, pde)
- `examples/pinn_forward/Euler_beam.py`: forward PINN example script: Euler_beam. (funcs: ddy, dddy, pde, boundary_l ...)
- `examples/pinn_forward/Helmholtz_Dirichlet_2d.py`: forward PINN example script: Helmholtz_Dirichlet_2d. (funcs: pde, func, transform, boundary)
- `examples/pinn_forward/Helmholtz_Dirichlet_2d_HPO.py`: forward PINN example script: Helmholtz_Dirichlet_2d_HPO. (funcs: pde, func, transform, boundary ...)
- `examples/pinn_forward/Helmholtz_Neumann_2d_hole.py`: forward PINN example script: Helmholtz_Neumann_2d_hole. (funcs: pde, func, boundary, neumann ...)
- `examples/pinn_forward/Helmholtz_Sound_hard_ABC_2d.py`: forward PINN example script: Helmholtz_Sound_hard_ABC_2d. (funcs: sound_hard_circle_deepxde, pde, sol, boundary ...)
- `examples/pinn_forward/Klein_Gordon.py`: forward PINN example script: Klein_Gordon. (funcs: pde, func)
- `examples/pinn_forward/Kovasznay_flow.py`: forward PINN example script: Kovasznay_flow. (funcs: pde, u_func, v_func, p_func ...)
- `examples/pinn_forward/Laplace_disk.py`: forward PINN example script: Laplace_disk. (funcs: pde, solution, feature_transform)
- `examples/pinn_forward/Lotka_Volterra.py`: forward PINN example script: Lotka_Volterra. (funcs: func, gen_truedata, ode_system, input_transform ...)
- `examples/pinn_forward/Poisson_Dirichlet_1d.py`: forward PINN example script: Poisson_Dirichlet_1d. (funcs: pde, boundary, func)
- `examples/pinn_forward/Poisson_Dirichlet_1d_exactBC.py`: forward PINN example script: Poisson_Dirichlet_1d_exactBC. (funcs: pde, func, output_transform)
- `examples/pinn_forward/Poisson_Lshape.py`: forward PINN example script: Poisson_Lshape. (funcs: pde, boundary)
- `examples/pinn_forward/Poisson_Neumann_1d.py`: forward PINN example script: Poisson_Neumann_1d. (funcs: pde, boundary_l, boundary_r, func)
- `examples/pinn_forward/Poisson_PointSetOperator_1d.py`: forward PINN example script: Poisson_PointSetOperator_1d. (funcs: pde, dy_x, boundary_l, func ...)
- `examples/pinn_forward/Poisson_Robin_1d.py`: forward PINN example script: Poisson_Robin_1d. (funcs: pde, boundary_l, boundary_r, func)
- `examples/pinn_forward/Poisson_multiscale_1d.py`: forward PINN example script: Poisson_multiscale_1d. (funcs: pde, func)
- `examples/pinn_forward/Poisson_periodic_1d.py`: forward PINN example script: Poisson_periodic_1d. (funcs: pde, boundary_l, boundary_r, func)
- `examples/pinn_forward/Schrodinger.ipynb`: Interactive notebook example: Schrodinger.
- `examples/pinn_forward/Volterra_IDE.py`: forward PINN example script: Volterra_IDE. (funcs: ide, kernel, func)
- `examples/pinn_forward/diffusion_1d.py`: forward PINN example script: diffusion_1d. (funcs: pde, func)
- `examples/pinn_forward/diffusion_1d_exactBC.py`: forward PINN example script: diffusion_1d_exactBC. (funcs: pde, func)
- `examples/pinn_forward/diffusion_1d_resample.py`: forward PINN example script: diffusion_1d_resample. (funcs: pde, func)
- `examples/pinn_forward/diffusion_reaction.py`: forward PINN example script: diffusion_reaction. (funcs: pde, func, output_transform)
- `examples/pinn_forward/elasticity_plate.py`: forward PINN example script: elasticity_plate. (funcs: boundary_left, boundary_right, boundary_top, boundary_bottom ...)
- `examples/pinn_forward/fractional_Poisson_1d.py`: forward PINN example script: fractional_Poisson_1d. (funcs: fpde, func)
- `examples/pinn_forward/fractional_Poisson_2d.py`: forward PINN example script: fractional_Poisson_2d. (funcs: fpde, func)
- `examples/pinn_forward/fractional_Poisson_3d.py`: forward PINN example script: fractional_Poisson_3d. (funcs: fpde, func)
- `examples/pinn_forward/fractional_diffusion_1d.py`: forward PINN example script: fractional_diffusion_1d. (funcs: fpde, func)
- `examples/pinn_forward/heat.py`: forward PINN example script: heat. (funcs: heat_eq_exact_solution, gen_exact_solution, gen_testdata, pde)
- `examples/pinn_forward/heat_resample.py`: forward PINN example script: heat_resample. (funcs: heat_eq_exact_solution, gen_exact_solution, gen_testdata, pde)
- `examples/pinn_forward/ide.py`: forward PINN example script: ide. (funcs: ide, func)
- `examples/pinn_forward/loss.dat`: pinn_forward example auxiliary data/output file.
- `examples/pinn_forward/ode_2nd.py`: forward PINN example script: ode_2nd. (funcs: ode, func, boundary_l, bc_func1 ...)
- `examples/pinn_forward/ode_system.py`: forward PINN example script: ode_system. (funcs: ode_system, boundary, func)
- `examples/pinn_forward/test.dat`: pinn_forward example auxiliary data/output file.
- `examples/pinn_forward/train.dat`: pinn_forward example auxiliary data/output file.
- `examples/pinn_forward/wave_1d.py`: forward PINN example script: wave_1d. (funcs: get_initial_loss, pde, func)
- `examples/pinn_inverse/Lorenz_inverse.py`: inverse PINN example script: Lorenz_inverse. (funcs: gen_traindata, Lorenz_system, boundary)
- `examples/pinn_inverse/Lorenz_inverse_forced.py`: inverse PINN example script: Lorenz_inverse_forced. (funcs: ex_func, LorezODE, ex_func2, Lorenz_system ...)
- `examples/pinn_inverse/Navier_Stokes_inverse.py`: inverse PINN example script: Navier_Stokes_inverse. (funcs: load_training_data, Navier_Stokes_Equation)
- `examples/pinn_inverse/brinkman_forchheimer.py`: inverse PINN example script: brinkman_forchheimer. (funcs: sol, gen_traindata, pde, output_transform)
- `examples/pinn_inverse/diffusion_1d_inverse.py`: inverse PINN example script: diffusion_1d_inverse. (funcs: pde, func)
- `examples/pinn_inverse/diffusion_reaction_rate.py`: inverse PINN example script: diffusion_reaction_rate. (funcs: k, fun, bc, gen_traindata ...)
- `examples/pinn_inverse/elliptic_inverse_field.py`: inverse PINN example script: elliptic_inverse_field. (funcs: gen_traindata, pde, sol)
- `examples/pinn_inverse/elliptic_inverse_field_batch.py`: inverse PINN example script: elliptic_inverse_field_batch. (funcs: gen_traindata, pde, sol)
- `examples/pinn_inverse/fractional_Poisson_1d_inverse.py`: inverse PINN example script: fractional_Poisson_1d_inverse. (funcs: fpde, func)
- `examples/pinn_inverse/fractional_Poisson_2d_inverse.py`: inverse PINN example script: fractional_Poisson_2d_inverse. (funcs: fpde, func)
- `examples/pinn_inverse/reaction_inverse.py`: inverse PINN example script: reaction_inverse. (funcs: gen_traindata, pde, fun_bc, fun_init)
- `examples/sample_to_test.py`: Transformer that rewrites example scripts into short integration-test form. (funcs: transform)

### myproject

- `myproject/README.md`: myproject subproject README (how to run with .mat data).
- `myproject/WORK_SUMMARY.md`: myproject modeling/training summary document.
- `myproject/calculate_drag_coefficient.m`: MATLAB baseline implementation for drag/force calculations.
- `myproject/check_data.py`: .mat structure inspection utility. (funcs: check_mat_file)
- `myproject/config.py`: Runtime environment setup helper (backend/path setup). (funcs: setup_environment)
- `myproject/extract_data.py`: .mat field extraction and introspection utility. (funcs: summarize_array, extract_data_from_mat)
- `myproject/my_euler_beam.py`: Euler-Bernoulli beam PINN experiment script with optional observed data points. (funcs: load_function, ddy, dddy, pde ...)
- `myproject/pinn_training_data.mat`: Primary myproject training dataset.
- `myproject/run_in_ide.py`: IDE-friendly launcher wrapper script.
- `myproject/runs_force/20250821-141226__E-100000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/cd_vs_re.png`: Experiment artifact (20250821-141226__E-100000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Cd vs Reynolds number plot.
- `myproject/runs_force/20250821-141226__E-100000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/console.log`: Experiment artifact (20250821-141226__E-100000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run stdout log.
- `myproject/runs_force/20250821-141226__E-100000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/error_vs_velocity.png`: Experiment artifact (20250821-141226__E-100000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Relative error vs velocity plot.
- `myproject/runs_force/20250821-141226__E-100000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/learned_cd_params.json`: Experiment artifact (20250821-141226__E-100000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Learned drag-parameter summary.
- `myproject/runs_force/20250821-141226__E-100000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/learned_cd_values.csv`: Experiment artifact (20250821-141226__E-100000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-sample exported values (Re, Cd, prediction decomposition, etc.).
- `myproject/runs_force/20250821-141226__E-100000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/pred_vs_true_velocity.png`: Experiment artifact (20250821-141226__E-100000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Prediction vs truth plot across inflow velocity.
- `myproject/runs_force/20250821-141226__E-100000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/run_config.json`: Experiment artifact (20250821-141226__E-100000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run configuration snapshot (hyperparameters and filters).
- `myproject/runs_force/20250821-141226__E-100000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/stderr.log`: Experiment artifact (20250821-141226__E-100000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run stderr log.
- `myproject/runs_force/20250821-141440__E-100000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/cd_vs_re.png`: Experiment artifact (20250821-141440__E-100000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Cd vs Reynolds number plot.
- `myproject/runs_force/20250821-141440__E-100000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/console.log`: Experiment artifact (20250821-141440__E-100000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run stdout log.
- `myproject/runs_force/20250821-141440__E-100000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/error_vs_velocity.png`: Experiment artifact (20250821-141440__E-100000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Relative error vs velocity plot.
- `myproject/runs_force/20250821-141440__E-100000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/learned_cd_params.json`: Experiment artifact (20250821-141440__E-100000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Learned drag-parameter summary.
- `myproject/runs_force/20250821-141440__E-100000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/learned_cd_values.csv`: Experiment artifact (20250821-141440__E-100000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-sample exported values (Re, Cd, prediction decomposition, etc.).
- `myproject/runs_force/20250821-141440__E-100000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/pred_vs_true_velocity.png`: Experiment artifact (20250821-141440__E-100000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Prediction vs truth plot across inflow velocity.
- `myproject/runs_force/20250821-141440__E-100000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/run_config.json`: Experiment artifact (20250821-141440__E-100000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run configuration snapshot (hyperparameters and filters).
- `myproject/runs_force/20250821-141440__E-100000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/stderr.log`: Experiment artifact (20250821-141440__E-100000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run stderr log.
- `myproject/runs_force/20250821-141851__E-2000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/cd_vs_re.png`: Experiment artifact (20250821-141851__E-2000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Cd vs Reynolds number plot.
- `myproject/runs_force/20250821-141851__E-2000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/console.log`: Experiment artifact (20250821-141851__E-2000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run stdout log.
- `myproject/runs_force/20250821-141851__E-2000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/error_vs_velocity.png`: Experiment artifact (20250821-141851__E-2000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Relative error vs velocity plot.
- `myproject/runs_force/20250821-141851__E-2000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/learned_cd_params.json`: Experiment artifact (20250821-141851__E-2000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Learned drag-parameter summary.
- `myproject/runs_force/20250821-141851__E-2000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/learned_cd_values.csv`: Experiment artifact (20250821-141851__E-2000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-sample exported values (Re, Cd, prediction decomposition, etc.).
- `myproject/runs_force/20250821-141851__E-2000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/pred_vs_true_velocity.png`: Experiment artifact (20250821-141851__E-2000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Prediction vs truth plot across inflow velocity.
- `myproject/runs_force/20250821-141851__E-2000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/run_config.json`: Experiment artifact (20250821-141851__E-2000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run configuration snapshot (hyperparameters and filters).
- `myproject/runs_force/20250821-141851__E-2000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/stderr.log`: Experiment artifact (20250821-141851__E-2000000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run stderr log.
- `myproject/runs_force/20250821-142631__E-2000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/cd_vs_re.png`: Experiment artifact (20250821-142631__E-2000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Cd vs Reynolds number plot.
- `myproject/runs_force/20250821-142631__E-2000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/console.log`: Experiment artifact (20250821-142631__E-2000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run stdout log.
- `myproject/runs_force/20250821-142631__E-2000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/error_vs_velocity.png`: Experiment artifact (20250821-142631__E-2000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Relative error vs velocity plot.
- `myproject/runs_force/20250821-142631__E-2000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/learned_cd_params.json`: Experiment artifact (20250821-142631__E-2000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Learned drag-parameter summary.
- `myproject/runs_force/20250821-142631__E-2000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/learned_cd_values.csv`: Experiment artifact (20250821-142631__E-2000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-sample exported values (Re, Cd, prediction decomposition, etc.).
- `myproject/runs_force/20250821-142631__E-2000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/pred_vs_true_velocity.png`: Experiment artifact (20250821-142631__E-2000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Prediction vs truth plot across inflow velocity.
- `myproject/runs_force/20250821-142631__E-2000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/run_config.json`: Experiment artifact (20250821-142631__E-2000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run configuration snapshot (hyperparameters and filters).
- `myproject/runs_force/20250821-142631__E-2000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/stderr.log`: Experiment artifact (20250821-142631__E-2000000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run stderr log.
- `myproject/runs_force/20250821-142855__E-300000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/cd_vs_re.png`: Experiment artifact (20250821-142855__E-300000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Cd vs Reynolds number plot.
- `myproject/runs_force/20250821-142855__E-300000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/console.log`: Experiment artifact (20250821-142855__E-300000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run stdout log.
- `myproject/runs_force/20250821-142855__E-300000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/error_vs_velocity.png`: Experiment artifact (20250821-142855__E-300000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Relative error vs velocity plot.
- `myproject/runs_force/20250821-142855__E-300000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/learned_cd_params.json`: Experiment artifact (20250821-142855__E-300000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Learned drag-parameter summary.
- `myproject/runs_force/20250821-142855__E-300000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/learned_cd_values.csv`: Experiment artifact (20250821-142855__E-300000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-sample exported values (Re, Cd, prediction decomposition, etc.).
- `myproject/runs_force/20250821-142855__E-300000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/pred_vs_true_velocity.png`: Experiment artifact (20250821-142855__E-300000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Prediction vs truth plot across inflow velocity.
- `myproject/runs_force/20250821-142855__E-300000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/run_config.json`: Experiment artifact (20250821-142855__E-300000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run configuration snapshot (hyperparameters and filters).
- `myproject/runs_force/20250821-142855__E-300000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/stderr.log`: Experiment artifact (20250821-142855__E-300000__h-0p02__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run stderr log.
- `myproject/runs_force/20250821-143207__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/cd_vs_re.png`: Experiment artifact (20250821-143207__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Cd vs Reynolds number plot.
- `myproject/runs_force/20250821-143207__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/console.log`: Experiment artifact (20250821-143207__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run stdout log.
- `myproject/runs_force/20250821-143207__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/error_vs_velocity.png`: Experiment artifact (20250821-143207__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Relative error vs velocity plot.
- `myproject/runs_force/20250821-143207__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/learned_cd_params.json`: Experiment artifact (20250821-143207__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Learned drag-parameter summary.
- `myproject/runs_force/20250821-143207__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/learned_cd_values.csv`: Experiment artifact (20250821-143207__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-sample exported values (Re, Cd, prediction decomposition, etc.).
- `myproject/runs_force/20250821-143207__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/pred_vs_true_velocity.png`: Experiment artifact (20250821-143207__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Prediction vs truth plot across inflow velocity.
- `myproject/runs_force/20250821-143207__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/run_config.json`: Experiment artifact (20250821-143207__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run configuration snapshot (hyperparameters and filters).
- `myproject/runs_force/20250821-143207__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/stderr.log`: Experiment artifact (20250821-143207__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run stderr log.
- `myproject/runs_force/20250904-012919__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/cd_vs_re.png`: Experiment artifact (20250904-012919__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Cd vs Reynolds number plot.
- `myproject/runs_force/20250904-012919__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/console.log`: Experiment artifact (20250904-012919__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run stdout log.
- `myproject/runs_force/20250904-012919__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/error_vs_velocity.png`: Experiment artifact (20250904-012919__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Relative error vs velocity plot.
- `myproject/runs_force/20250904-012919__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/learned_cd_params.json`: Experiment artifact (20250904-012919__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Learned drag-parameter summary.
- `myproject/runs_force/20250904-012919__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/learned_cd_values.csv`: Experiment artifact (20250904-012919__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-sample exported values (Re, Cd, prediction decomposition, etc.).
- `myproject/runs_force/20250904-012919__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/pred_vs_true_velocity.png`: Experiment artifact (20250904-012919__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Prediction vs truth plot across inflow velocity.
- `myproject/runs_force/20250904-012919__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/run_config.json`: Experiment artifact (20250904-012919__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run configuration snapshot (hyperparameters and filters).
- `myproject/runs_force/20250904-012919__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/stderr.log`: Experiment artifact (20250904-012919__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run stderr log.
- `myproject/runs_force/20260413-181751__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-200__tol-1e-06/cd_vs_re.png`: Experiment artifact (20260413-181751__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-200__tol-1e-06): Cd vs Reynolds number plot.
- `myproject/runs_force/20260413-181751__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-200__tol-1e-06/console.log`: Experiment artifact (20260413-181751__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-200__tol-1e-06): Per-run stdout log.
- `myproject/runs_force/20260413-181751__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-200__tol-1e-06/error_vs_velocity.png`: Experiment artifact (20260413-181751__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-200__tol-1e-06): Relative error vs velocity plot.
- `myproject/runs_force/20260413-181751__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-200__tol-1e-06/learned_cd_params.json`: Experiment artifact (20260413-181751__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-200__tol-1e-06): Learned drag-parameter summary.
- `myproject/runs_force/20260413-181751__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-200__tol-1e-06/learned_cd_values.csv`: Experiment artifact (20260413-181751__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-200__tol-1e-06): Per-sample exported values (Re, Cd, prediction decomposition, etc.).
- `myproject/runs_force/20260413-181751__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-200__tol-1e-06/pred_vs_true_velocity.png`: Experiment artifact (20260413-181751__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-200__tol-1e-06): Prediction vs truth plot across inflow velocity.
- `myproject/runs_force/20260413-181751__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-200__tol-1e-06/run_config.json`: Experiment artifact (20260413-181751__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-200__tol-1e-06): Per-run configuration snapshot (hyperparameters and filters).
- `myproject/runs_force/20260413-181751__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-200__tol-1e-06/stderr.log`: Experiment artifact (20260413-181751__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-200__tol-1e-06): Per-run stderr log.
- `myproject/runs_force/20260413-183845__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/cd_vs_re.png`: Experiment artifact (20260413-183845__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Cd vs Reynolds number plot.
- `myproject/runs_force/20260413-183845__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/console.log`: Experiment artifact (20260413-183845__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run stdout log.
- `myproject/runs_force/20260413-183845__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/error_vs_velocity.png`: Experiment artifact (20260413-183845__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Relative error vs velocity plot.
- `myproject/runs_force/20260413-183845__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/learned_cd_params.json`: Experiment artifact (20260413-183845__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Learned drag-parameter summary.
- `myproject/runs_force/20260413-183845__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/learned_cd_values.csv`: Experiment artifact (20260413-183845__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-sample exported values (Re, Cd, prediction decomposition, etc.).
- `myproject/runs_force/20260413-183845__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/pred_vs_true_velocity.png`: Experiment artifact (20260413-183845__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Prediction vs truth plot across inflow velocity.
- `myproject/runs_force/20260413-183845__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/run_config.json`: Experiment artifact (20260413-183845__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run configuration snapshot (hyperparameters and filters).
- `myproject/runs_force/20260413-183845__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08/stderr.log`: Experiment artifact (20260413-183845__E-300000__h-0p01__Hc-None__ang-None__AM-max__iter-1000__tol-1e-08): Per-run stderr log.
- `myproject/runs_force/LATEST.txt`: Experiment artifact (runs_force): Pointer to latest run directory name.
- `myproject/test.dat`: myproject test output sample.
- `myproject/train.dat`: myproject training output sample.
- `myproject/train_force_model.py`: Main training pipeline: physics-prior baseline + residual net + Cd(Re) learning. (classes: ResidualMLP; funcs: load_dataset, finite_difference_matrix, compute_force_matlab_style, compute_physics_priors ...)
- `myproject/train_run.log`: Historical training log file.
- `myproject/train_run_forcepos.log`: Historical training log file (forcepos variant).

### docker

- `docker/Dockerfile`: Docker image definition (Horovod-based, notebook-capable DeepXDE environment).
- `docker/requirements.txt`: Python dependency list for Docker environment.

## 5. Suggested Reading Order

1. `README.md`, `pyproject.toml`, `deepxde/model.py`
2. `deepxde/data/pde.py` and `deepxde/icbc/boundary_conditions.py`
3. `examples/` by topic (`pinn_forward`, `pinn_inverse`, `operator`)
4. `myproject/train_force_model.py` and `myproject/WORK_SUMMARY.md`

## 6. One-Sentence Summary

This repo is a production-grade scientific ML framework plus a complete set of demos, extended with a reproducible custom physics-informed engineering workflow in `myproject`.