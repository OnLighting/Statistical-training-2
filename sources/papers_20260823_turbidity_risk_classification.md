# 问题四：浊度风险分类依据检索记录

检索日期：2026-08-23

## 结论

未发现可直接照搬的“按日、仅依据出厂水浊度超标幅度与持续时长划分安全/低/中/高风险”的统一国家或国际标准。可采用“强制限值 + 风险矩阵 + 超标幅度/频率/持续时间指标”的二次构造，并明确区分法定阈值与研究者设定阈值。

## 主要依据

1. 国家市场监督管理总局、国家标准化管理委员会：GB 5749-2022《生活饮用水卫生标准》，现行强制性国家标准。题目指定浑浊度 1 NTU 为统一硬约束。
   - https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=99E9C17E3547A3C0CE2FD1FFD9F2F7BE&refer=outter

2. WHO, *Water safety plan manual: step-by-step risk management for drinking-water suppliers*, 2nd ed., 2023. WHO 推荐按发生可能性与后果严重度构造风险矩阵，并强调风险定义应结合具体供水系统预先明确，而非使用全球统一分级阈值。
   - https://www.who.int/publications/i/item/9789240067691

3. WHO, *Water quality and health: Review of turbidity*, 2017. 浑浊度是水处理运行和潜在危险事件的重要指示量；小于 1 NTU 有利于有效消毒，浑浊度本身并非在所有情境下都可直接等同健康风险。
   - https://www.who.int/publications/i/item/WHO-FWC-WSH-17.01

4. CCME, *Synthesis of Research and Application of the CCME Water Quality Index*, 2017. CCME WQI 用 scope、frequency、amplitude 三个分量概括超标范围、频率和幅度，并给出质量等级；同时指出指数对参数数目、采样数及时间窗敏感，建议使用足够多且相关的参数。因此原指数不宜直接用于“单一 NTU 指标的一日 12 点”评价，但其频率和幅度思想可以借鉴。
   - https://ccme.ca/en/res/synthesis-of-research-and-application-of-the-ccme-water-quality-index-2017.pdf

5. US EPA, *National Primary Drinking Water Regulations*. 对常规或直接过滤系统，代表性样本不得在任何时刻超过 1 NTU，且每月至少 95% 的样本不超过 0.3 NTU。该规定针对美国过滤处理技术合规，不等同于中国出厂水逐日四级风险标准，可作为敏感性分析的更严格参照方案。
   - https://www.epa.gov/ground-water-and-drinking-water/national-primary-drinking-water-regulations

6. Muoio R, Caretti C, Rossi L, Santianni D, Lubello C. Water safety plans and risk assessment: A novel procedure applied to treated water turbidity and gastrointestinal diseases. *International Journal of Hygiene and Environmental Health*. 2020;223(1):281-288. DOI: 10.1016/j.ijheh.2019.07.008. 论文表明可使用处理后水浊度时间序列量化相对健康风险，但没有给出可普遍照搬的四级日风险阈值。
   - https://pubmed.ncbi.nlm.nih.gov/31523016/

7. De Marines F, et al. A novel methodological approach for the assessment of drinking water treatment plant robustness under challenging turbidity scenarios. *Journal of Water Process Engineering*. 2025;75:107971. DOI: 10.1016/j.jwpe.2025.107971. ALERT 方法使用分位数稳健性指标、浊度负荷和短时极端事件指标，支持把峰值以外的累计负荷和事件特征纳入评价。
   - https://doi.org/10.1016/j.jwpe.2025.107971

8. De Roos AJ, et al. Review of Epidemiological Studies of Drinking-Water Turbidity in Relation to Acute Gastrointestinal Illness. *Environmental Health Perspectives*. 2017;125(8):086004. DOI: 10.1289/EHP1090. 综述认为浑浊度与急性胃肠道疾病的关联具有情境依赖性，建议结合季节、气候、其他水质指标和处理过程数据解释。
   - https://pubmed.ncbi.nlm.nih.gov/28886603/

## 对本题的建议

- 法定硬约束：任一有效时点 NTU > 1，即当日不得归为“安全”。
- 日风险特征：峰值超标倍数、超标观测比例、最长连续超标时长、累计超标负荷（NTU·h）。
- 主模型：借鉴 WHO 的“严重度 × 持续/频率”风险矩阵，并用模糊隶属函数减少硬边界跳变。
- 对照模型：保留评分参考给出的透明阈值法，报告两种方法的一致率或 Cohen's kappa。
- 稳健性：改变幅度、时长阈值并比较等级占比；另以 EPA 的 0.3 NTU 月度运行目标作为严格情景，但不得称为中国法定标准。
