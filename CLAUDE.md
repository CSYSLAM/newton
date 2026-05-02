@AGENTS.md

## 运行示例（Windows PowerShell）

```powershell
# 1. 进入 newton 根目录
cd c:\csy_work\CG\Engine\newton

# 2. 设置执行策略并激活虚拟环境
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& c:\csy_work\CG\Engine\newton\.venv\Scripts\Activate.ps1)

# 3. 运行 demo，例如：
python -m newton.examples basic_pendulum
```
