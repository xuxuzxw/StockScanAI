# 系统检查指南

## 概述

`run_system_check.py` 是A股量化投研平台的统一系统健康检查工具，整合了所有测试和性能检查功能。

## 使用方法

### 基本命令

```bash
# 完整系统检查（推荐）
python run_system_check.py

# 快速检查（仅基础功能）
python run_system_check.py --mode quick

# 稳定性检查（专注数据质量）
python run_system_check.py --mode stability

# 性能检查（API和数据库性能）
python run_system_check.py --mode performance
```

## 检查模式说明

### 1. 完整检查 (full)
- **检查项目**: 10项全面检查
- **包含内容**: 
  - 配置文件验证
  - 数据库连接测试
  - Tushare API连通性
  - 系统资源状态
  - 数据库性能
  - 数据新鲜度
  - 财务数据质量
  - 因子计算功能
  - AI分析层
  - 端到端工作流
- **适用场景**: 系统部署后的全面验证

### 2. 快速检查 (quick)
- **检查项目**: 4项基础检查
- **包含内容**:
  - 配置文件验证
  - 数据库连接
  - API连通性
  - 基础因子计算
- **适用场景**: 日常快速验证

### 3. 稳定性检查 (stability)
- **检查项目**: 6项稳定性检查
- **包含内容**:
  - 基础功能验证
  - 系统资源监控
  - 数据库性能
  - 数据存储测试
- **适用场景**: 系统稳定性评估

### 4. 性能检查 (performance)
- **检查项目**: 5项性能检查
- **包含内容**:
  - API并发性能测试
  - 查询性能基准测试
  - 数据库性能分析
- **适用场景**: 性能优化验证

## 检查结果解读

### 成功标识
- `[PASS]` - 检查通过
- `✓` - 功能正常

### 警告标识
- `[WARN]` - 需要关注但不影响运行
- `⚠` - 性能或配置建议

### 错误标识
- `[FAIL]` - 检查失败，需要修复
- `❌` - 功能异常

## 常见问题解决

### 1. 配置问题
```
[FAIL] 配置检查失败: TUSHARE_TOKEN 未配置
```
**解决方案**: 检查 `.env` 文件中的配置项

### 2. 数据库连接问题
```
[FAIL] 数据库连接失败
```
**解决方案**: 
- 确认数据库服务运行
- 检查连接配置
- 验证网络连通性

### 3. API限制问题
```
⚠ API并发性能: 需要关注
```
**解决方案**: 
- 检查Tushare积分余额
- 降低并发请求数量
- 考虑升级API套餐

### 4. 性能问题
```
⚠ 性能超时 (期望<200ms)
```
**解决方案**:
- 运行数据库优化: `python quick_optimize.py`
- 检查系统资源使用情况
- 考虑添加更多索引

## 维护建议

### 日常检查
- 每日运行快速检查确保基础功能正常
- 每周运行完整检查进行全面验证

### 性能监控
- 定期运行性能检查评估系统状态
- 在数据量增长后重新评估性能基准

### 数据质量
- 定期运行稳定性检查确保数据完整性
- 关注数据新鲜度警告及时更新数据

## 相关工具

- `quick_optimize.py` - 数据库一键优化
- `optimize_database.py` - 深度数据库优化分析
- `database_final_optimizer.py` - 综合数据库优化方案

## 版本历史

- **V3.3**: 整合所有测试功能，删除重复文件
- **V2.5**: 增加数据质量检查
- **V2.0**: 基础系统检查功能