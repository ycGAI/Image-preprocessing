图像清晰度分类工具
这是一个基于传统计算机视觉算法的图像清晰度分类工具，无需深度学习训练即可对图像进行清晰度评估和分类。

功能特点
🚀 免训练: 使用传统CV算法，无需深度学习模型训练
📊 多算法集成: 支持拉普拉斯、Sobel、Brenner、Tenengrad等多种清晰度评估算法
🔄 批量处理: 支持大批量图像的并行处理
📁 智能文件管理: 自动识别时间格式文件夹，处理图像-JSON文件对
📈 详细报告: 生成处理报告、可视化图表和CSV导出
🛠 灵活配置: 支持配置文件和命令行参数
🖥 交互模式: 提供用户友好的交互式运行模式
项目结构
image_sharpness_classifier/
├── sharpness_classifier.py    # 核心清晰度分类算法
├── file_utils.py             # 文件处理工具
├── image_processor.py        # 图像处理器
├── batch_processor.py        # 批量处理器
├── report_generator.py       # 报告生成器
├── main.py                   # 主程序入口
├── config.json              # 配置文件模板
├── requirements.txt         # 依赖文件
└── README.md               # 说明文档
安装依赖
bash
pip install -r requirements.txt
核心依赖
opencv-python: 图像处理
numpy: 数值计算
scipy: 科学计算
tqdm: 进度条显示
可选依赖（用于报告生成）
matplotlib: 图表生成
seaborn: 统计图表
pandas: 数据处理和CSV导出
使用方法
1. 命令行模式
基本使用
bash
python main.py --source /path/to/source --output /path/to/output
使用配置文件
bash
python main.py --config config.json
预览模式
bash
python main.py --preview --source /path/to/source --max-folders 3
创建配置文件模板
bash
python main.py --create-config my_config.json
2. 交互模式
直接运行主程序，按提示输入参数：

bash
python main.py
3. 高级选项
bash
python main.py \
  --source ./images \
  --output ./results \
  --max-workers 8 \
  --log-level DEBUG \
  --log-file processing.log \
  --no-visual \
  --no-csv
配置文件
创建 config.json 文件来自定义处理参数：

json
{
  "source_root": "/path/to/source",
  "output_root": "/path/to/output",
  "classifier_params": {
    "laplacian_threshold": 100.0,
    "sobel_threshold": 50.0,
    "brenner_threshold": 1000.0,
    "tenengrad_threshold": 500.0
  },
  "processing": {
    "max_workers": 4,
    "supported_formats": [".jpg", ".png", ".bmp"]
  },
  "reports": {
    "generate_visual": true,
    "export_csv": true
  }
}
算法说明
支持的清晰度评估算法
拉普拉斯方差 (Laplacian Variance)
使用拉普拉斯算子检测边缘变化
适用于大部分自然图像
Sobel算子方差 (Sobel Variance)
基于Sobel边缘检测算子
对噪声较为鲁棒
Brenner梯度 (Brenner Gradient)
计算水平方向梯度的平方和
计算速度快
Tenengrad算子 (Tenengrad)
改进的梯度方法
设置阈值过滤低梯度区域
集成分类策略
工具使用投票机制进行最终分类：

每个算法根据其阈值进行二分类判断
超过半数算法认为图像清晰时，最终判定为清晰
否则判定为模糊
输入文件结构
工具期望的输入文件结构：

source_folder/
├── 2024-01-01/          # 时间格式文件夹
│   ├── image001.jpg     # 图像文件
│   ├── image001.json    # 对应的JSON元数据
│   ├── image002.jpg
│   └── image002.json
├── 2024-01-02/
│   ├── photo001.png
│   ├── photo001.json
│   └── ...
└── ...
支持的时间格式
2024-01-01 (YYYY-MM-DD)
2024_01_01 (YYYY_MM_DD)
20240101 (YYYYMMDD)
2024-01-01_12-30-45 (YYYY-MM-DD_HH-MM-SS)
20240101123045 (YYYYMMDDHHMMSS)
输出结构
output_folder/
├── 2024-01-01/
│   ├── 清晰/             # 清晰图像及其JSON文件
│   │   ├── image001.jpg
│   │   └── image001.json
│   └── 模糊/             # 模糊图像及其JSON文件
│       ├── image002.jpg
│       └── image002.json
├── processing_report.json      # 处理报告
├── detailed_analysis.json      # 详细分析
├── processing_results.csv      # CSV格式结果
├── quality_distribution.png    # 质量分布图
├── folder_distribution.png     # 文件夹分布图
└── quality_ratio_histogram.png # 清晰度比例直方图
阈值调优
不同类型的图像可能需要不同的阈值设置：

自然风景照片
json
{
  "laplacian_threshold": 100.0,
  "sobel_threshold": 50.0,
  "brenner_threshold": 1000.0,
  "tenengrad_threshold": 500.0
}
人像照片
json
{
  "laplacian_threshold": 80.0,
  "sobel_threshold": 40.0,
  "brenner_threshold": 800.0,
  "tenengrad_threshold": 400.0
}
文档/文字图像
json
{
  "laplacian_threshold": 150.0,
  "sobel_threshold": 70.0,
  "brenner_threshold": 1500.0,
  "tenengrad_threshold": 700.0
}
性能优化
提高处理速度
增加线程数: 根据CPU核心数调整 max_workers
减少算法: 如果对精度要求不高，可以只使用单一算法
批量大小: 处理大量小文件时，适当增加线程数
内存优化
降低线程数: 处理大图像时减少并发数
图像格式: 优先使用压缩格式减少内存占用
常见问题
Q: 如何处理非时间格式的文件夹？
A: 可以修改 file_utils.py 中的 is_time_format 方法，或使用 run_with_custom_filter 方法自定义文件夹过滤逻辑。

Q: 如何调整分类精度？
A: 通过调整各算法的阈值参数，可以提高或降低分类严格程度。较高的阈值会导致更少的图像被判定为清晰。

Q: 支持哪些图像格式？
A: 默认支持 JPG、PNG、BMP、TIFF、WebP 格式，可在配置文件中修改。

Q: 如何处理大批量图像？
A: 建议使用预览模式先测试小批量，确认效果后再进行完整处理。可以适当增加线程数加速处理。

扩展开发
添加新的清晰度算法
在 sharpness_classifier.py 中添加新方法：

python
def your_custom_method(self, image: np.ndarray) -> float:
    # 实现你的算法
    return metric_value
自定义文件处理逻辑
继承并扩展 FileUtils 类：

python
class CustomFileUtils(FileUtils):
    def find_custom_file_pairs(self, folder: Path):
        # 实现自定义文件匹配逻辑
        pass
许可证
MIT License

贡献
欢迎提交 Issue 和 Pull Request！

更新日志
v1.0.0
初始版本发布
支持多种清晰度评估算法
批量处理和报告生成功能
命令行和交互模式支持
