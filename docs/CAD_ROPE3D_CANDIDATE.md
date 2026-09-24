# CAD读取中的3D RoPE候选

状态：已实现并通过CPU功能/编译检查；未进行GPU性能、训练收益或位姿精度验证。现有25000步训练仍使用固定V16执行快照。本模块默认不启用。

## 接入位置

仅在JEPA第2/4个block之前的SurfaceRead（代码layer索引1/3）对patch query与256个静态CAD表面token的Q/K施加XYZ旋转。它影响统一patch latent，再由同一latent恢复特征/几何并进入object query。没有并行位姿路径。保留DINO本身的位置编码、现有2D图像位置、几何MLP、UV软先验、state token和显式残差。global object/state/null token没有可靠的3D位置，不强行旋转。

观测patch使用实测深度汇聚的XYZ。现有obs.object_xyz已经按当前估计位姿逆变换并按直径归一化；将它与CAD canonical XYZ同时乘当前估计R，得到相机轴方向、以当前估计物体中心为共同原点的坐标：q=(Xobs-tbase)/d，k=Rbase Xcad/d。两者之差保留当前估计位姿误差。只用当前估计，不使用GT或预测补全XYZ。

每头32维：XYZ各占10维（5对旋转通道），剩余2维保留；频率0.5/1/2/4/8 cycles per diameter。新增4个零初始化门控参数：logits=old+tanh(gain)*detached_visibility_confidence*valid*(rotary-old)。门控0时原前向逐值不变，门控仍有梯度。轴向RoPE不保证SO(3)旋转不变性。

缺深度或无观测置信度时，位置残差严格为0，回退原attention；不能只给无深度query填0而单独旋转CAD keys。置信度取原visibility head对真实观测特征的预测并detach，不能通过这条loss梯度主动压低置信度。该机制不能保证把遮挡手的深度识别为错误物体深度；patch多表面混合及置信度误差仍是限制。真实重遮挡中没有可靠query XYZ时，不凭空补造坐标。

## 使用与迁移

构建配置可显式添加cad_rope3d.enabled=true及cycles_per_diameter=[0.5,1,2,4,8]；也可先严格载入旧模型，再调用enable_cad_rope3d(model)。此开关属于结构实验，不能通过V16 execution-only迁移冒充无语义变化的速度优化。

旧checkpoint迁移仅允许新增cad_surface.rope3d.gain；其他旧张量逐值保留。冻结crop参考模型显式移除此开关，避免改变crop协议。新版checkpoint包含4个门控参数，原CAD资产/特征缓存不失效。现有长跑没有自动切换，也没有追加实验预算。

## 验证

19项CPU检查通过（test_rope3d/test_cad_surface/test_reconstruction_only）：旋转范数与相对平移性质；零门控原模型一致；缺失/无置信度/丢弃CAD回退；坐标及置信度stop-gradient；新门控及统一JEPA梯度；fullgraph aot_eager编译；新增参数严格保存/恢复。aot_eager CPU检查不是H20实测。

后续应从同一checkpoint做有/无RoPE的等预算对照，比较局部检索、CAD表面对应、XYZ/depth、一观测多估计位姿响应及速度。当前没有证据证明它优于原模型或能单独改善位姿。

参考：[RoFormer](https://arxiv.org/abs/2104.09864)、[GeoLRM](https://linshan-bin.github.io/GeoLRM/static/GeoLRM_arXiv.pdf)。此处为本项目的置信度门控轴向XYZ设计，不是上述工作的完整复现。
