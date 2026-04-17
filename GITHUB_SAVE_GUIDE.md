# GitHub 保存说明

这份复现工程已经整理成可以直接上传的结构。

## 现在这份工程里有什么

- 论文全文抽取结果
- 论文结构整理结果
- 六组实验的可执行实现
- 训练、验证、测试数据
- 六组实验的结果文件和模型文件

## 上传前你需要知道的事

- 这是一个“带数据”的完整版本，不只是代码。
- 当前整个目录大约 `608 MB`。
- 其中最大头是 `artifacts/` 里的数据集。
- 这意味着第一次 `git push` 会比较久，但按当前文件大小看，正常 GitHub 仓库可以承受。

## 推荐上传方式

在这个目录里执行：

```powershell
cd D:\keyan\research-units-pipeline-skills-main\workspaces\open-set-laplacian-pyramid-repro
git init -b main
git add .
git commit -m "Initial full reproduction of open-set laplacian pyramid paper"
git remote add origin 你的仓库地址
git push -u origin main
```

如果你已经初始化过 Git，就从 `git add .` 开始就行。

## 推荐仓库名

`open-set-laplacian-pyramid-repro`

## 如果你想要更轻一点的版本

可以只上传这些核心内容：

- `repro_dataset.py`
- `repro_model.py`
- `run_repro.py`
- `README.md`
- `requirements.txt`
- `output/CLAIMS.md`
- `output/RECONSTRUCTION_NOTES.md`
- `output/reproduction_summary.md`

然后把大数据集和模型文件留在本地。

但如果你的目标是“完整复现可直接复跑”，建议保留当前完整版本。
