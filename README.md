# 数字人口播工作台

作者：Manny · MIT · 0.3.0 实验版

本数字人系统由 **Manny 与 Codex 协作完成**。Manny 负责需求、场景与验收反馈，Codex 辅助实现、排查、测试和文档整理；作者与维护者保留 Manny。项目不是 OpenAI 官方产品或背书项目，底层模型仍属于各自作者与社区。

面向单机创作者的本地数字人口播工作台：管理形象和声音、输入文案、串行生成带字幕的口播视频。前端为 React / Vinext，后端为 FastAPI / SQLite，编排 Qwen3-TTS、MuseTalk 1.5 和 FFmpeg。它不是自研生成模型，也不是可直接公网部署的多用户 SaaS。

## 功能

产品背景、协作声明和优化证据见 [产品介绍与真实改进说明](docs/PRODUCT.md)。工作台提供语义段配音、停顿与音量平滑、章节动作相位延续、完整音轨统一合成和中文词组字幕。v0.3.0 新增生成前移的质量检查、有限失败回退、ASR 辅助字幕定位与整数帧章节；修复 v0.2.1 曾将异常配音当成功的严重缺陷。详见 [质量检查与升级配置](docs/QUALITY.md)。这些是工作流改进，不代表重新训练或全面超越上游模型。

本次完整回归与限制见 [v0.3.0 发布说明](docs/RELEASE-0.3.0.md)。

- 形象、声音和成片管理；受管文件删除后移入回收站，外部引用文件不删除。
- 上传声音自动转为 24 kHz 单声道；参考视频限制在 720×1280 边界内并转为 25 fps。
- 单 GPU 队列、进度、错误提示、取消排队和失败重试。
- 连续音轨、分章节口型驱动、最终统一字幕和音画合成。
- 上传与队列限额、后台进程互斥、推理超时及最终成片原子替换。
- 每次重试使用独立目录，避免新配音混用旧口型视频。

## 安装边界

目前仅在 Ubuntu 24.04 / WSL2 + NVIDIA GPU 上验证。需要 Python 3.12（工作台后端）、Node.js 22.13+、FFmpeg、中文字体，以及单独安装的模型环境。CPU-only、macOS、原生 Windows 和其他 GPU 未验证。

工作台不包含模型权重，不会自动安装 CUDA，也不保证其他显卡直接可用。建议把正式项目放在 WSL 的 Linux 文件系统中，媒体目录可单独配置。

```bash
sudo apt install ffmpeg fonts-noto-cjk curl jq
npm ci
python3 -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements.txt
```

### 单独配置推理环境

下列相对路径示例假设源码目录名为 `avatar-studio`；解压后请重命名目录或相应调整命令。

请依据上游说明安装：

- [MuseTalk](https://github.com/TMElyralab/MuseTalk)：本次兼容性验证针对 commit `0a89dec45a0192b824e3cf4daf96c239440c5ed8`，使用 Python 3.10 的独立环境。
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)：调用 `Qwen/Qwen3-TTS-12Hz-0.6B-Base`，需要 qwen-tts 和 jieba；在该环境执行 `uv pip install --python /path/to/tts-python --no-deps -r pipeline/requirements-quality.txt` 追加中文质检依赖。不要与 MuseTalk 共用虚拟环境。
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp)：v0.3.0 必需的离线逐段质检，需配置 `whisper-cli` 和中文模型。已有安装可以复用，缺失时不会跳过检查。[配置说明](docs/QUALITY.md)

参考目录结构：

```text
projects/
  avatar-studio/          # 当前源码
  ai-avatar-local/
    MuseTalk/
      .venv/bin/python
      models/               # 按上游说明下载
    tts-qwen3/
      .venv/bin/python
```

模型首次使用可能下载权重；下载完成后推理在本地执行。模型安装路径和工作目录建议不含空格，以兼容上游命令调用。PyTorch / CUDA 版本需匹配各模型与显卡，请勿盲目统一升级。

对上述 MuseTalk 版本应用章节动作相位补丁：

```bash
git -C ../ai-avatar-local/MuseTalk apply --check ../../avatar-studio/patches/musetalk-phase.patch
git -C ../ai-avatar-local/MuseTalk apply ../../avatar-studio/patches/musetalk-phase.patch
```

若检查失败，先核对版本和已有修改，不要强制覆盖。新补丁是累计版本：已应用 v0.2.1 补丁的用户须按 [升级说明](docs/QUALITY.md) 检查反向应用旧补丁后再应用新版，不能直接叠加。补丁保留上游版权声明。

### 本地路径

复制 `studio.example.json` 为 `studio.local.json`，把示例占位路径替换为真实路径。此文件已加入忽略规则，不应公开。

- `AVATAR_STUDIO_MEDIA_ROOT`：素材、成片、回收站；默认 `runtime/media`。
- `AVATAR_STUDIO_RUNTIME`：数据库、日志和缓存；默认项目内 `runtime`。
- `AVATAR_STUDIO_AI_PROJECT`：默认相邻 `ai-avatar-local`。
- 也可分别设置 `AVATAR_STUDIO_TTS_PYTHON`、`AVATAR_STUDIO_MUSETALK`、`AVATAR_STUDIO_MUSETALK_PYTHON`。

同名环境变量优先于本地 JSON。不会自动导入机器上的个人素材。

### 启动

```bash
bash scripts/run-service.sh
```

打开 http://localhost:3000 。前后端只监听回环地址，后端端口为 8765。此入口在前台运行；关闭它会停止服务。

`scripts/start-studio.sh` 和 `scripts/keepalive.sh` 仅适用于已经安装了 `avatar-studio.service` 的机器，本源码包不自动安装 systemd 服务或 Windows 启动器。现阶段启动脚本使用前端开发服务器；构建通过不代表已提供生产部署方案。

## 使用

1. 上传本人或已获授权的人像视频，建议正面、清晰、固定机位。
2. 上传 8–20 秒干净的人声，填写与音频逐字对应的参考原文。
3. 选择形象和声音，输入文案，加入队列。
4. 在成片管理中播放、下载或查看错误。

上传文件最大 256 MB；参考视频最多 120 秒、4K，参考音频最多 60 秒。单篇文案最多 10,000 字符，最多 32 个活动任务。推理子进程默认六小时超时。运行中素材不可转码或删除。

本机一次历史长片测试：约 326 秒成片在 RTX 2060 上耗时约 44 分钟，仅供参考，不是其他设备或文案的性能保证。

## 已知限制

- 当前配音与质量检查默认中文；其他语言、方言和大量中英混读未充分验证，不能直接套用相同阈值。
- 字幕为 ASR 时间锚点辅助的近似对齐，不是逐字/音素强制对齐；识别错误仍可能造成局部提前或滞后。
- 配音检查和可选 SyncNet 评分会误拒绝或漏检；自动通过不等于主观质量保证，成片必须预览验收。
- 尚无情绪滑杆或分段情绪控制；参考音频会影响表现，但无法保证指定情绪。
- 固定帧率不等于消除模型抖动、重复动作或所有视觉卡顿。
- 重试会重新生成，不支持安全的断点续算；历史 attempt 缓存保留，需手动清理。
- 可取消排队任务；运行中任务目前只能通过停止服务中断。
- 成片列表只返回最近 100 条；无分页、自动缓存清理或图形化回收站恢复。
- 无账号认证、租户隔离或公网安全设计。不要把服务暴露到公网或局域网。
- 推理环境仍依赖上游旧版依赖；本项目后端审计不覆盖整套模型环境。

详见 [安全边界](SECURITY.md)、[第三方许可](THIRD_PARTY_NOTICES.md) 和 [本次审计](docs/AUDIT-2026-09-06.md)。

## 验证

```bash
backend/.venv/bin/python -m pip install -r backend/requirements-dev.txt
backend/.venv/bin/python -m unittest discover -s tests -v
/path/to/tts-python -m unittest discover -s tests -p test_quality.py -v
npm run lint
npm run build
npm audit
```

回归测试使用临时数据库和合成素材，不读写正式媒体。真实 GPU 冒烟测试需显式提供已授权素材：

```bash
backend/.venv/bin/python scripts/smoke-generation.py --avatar /path/to/avatar.mp4 --voice /path/to/reference.wav --ref-text '参考音频的逐字原文'
```

## 生成脱敏源码包

```bash
python3 scripts/export-source.py
```

输出 `release/avatar-studio-0.3.0-source.zip`。导出器按白名单收集文本源码、执行基础敏感信息检查，并附带 SHA-256 清单。它不包含 Git 历史、数据库、日志、参考素材、成片、虚拟环境、模型或本地配置。发布前仍需人工复核新增文件，自动扫描不是保密保证。

## 许可

原创工作台代码：Copyright (c) 2026 Manny，MIT。模型、第三方库、补丁所涉及的上游代码及媒体权利分别遵循原许可和授权，不能统一重新标为 MIT。使用者须确保声音、人像、文案与输出的使用合法且获得必要授权。

## 反馈

如果有任何问题，欢迎大家反馈：**[ningfangtianwai@gmail.com](mailto:ningfangtianwai@gmail.com)**。

反馈时请提供版本、系统/显卡、复现步骤和经过脱敏的错误信息，不要发送密钥、私人文案、数据库或未经授权的人像与声音。
