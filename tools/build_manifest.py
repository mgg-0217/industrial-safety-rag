"""语料清单生成器：把 data/raw + data/md 的真实文档汇总成 manifest.json + 可读清单。

元数据字段刻意对齐 PRD 里要求的 schema（见 docs/PRD.md 第 2 节）：
  文档级  doc_id / title / doc_type / discipline / version / effective_date / status / classification / source_url
  条款级  hazard_tags / equipment_tags / maintenance_type / ref_standard
标注值全部来自文档本身（标题、发文号、落款），不臆造。

注：枚举取值以本文件实际产出为准（extract_quality 的扫描件取值为 `ocr_required`），
docs/PRD.md 附录 B 记录了与早期 PRD 生成提示词之间的 4 处实测差异。
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MD = ROOT / "data" / "md"

#: 把**项目根**放进 sys.path —— `src/versioning.py` 要按包导入
#: （`from src.versioning import ...`）。
#: 🔴 只把 `src` 放进 path 是不够的：那样 `src.versioning` 导不进来，
#: 而 `build_manifest.py` 是从 `tools/` 目录启动的（sys.path[0] = tools/），
#: 根目录不在里面 —— 于是版本字段补算这一步被**静默跳过**，
#: manifest 被写成一份**没有 effective_from / effective_to / superseded_by** 的版本。
#: README 里"重跑数据流水线"第一条就是本脚本，静默丢字段比报错危险得多。
sys.path.insert(0, str(ROOT))

# ---------------- 人工标注（依据文档标题/发文号/落款逐份核对） ----------------
ANNOTATIONS = {
    "yulin_donghuo_xianshi": {
        "title": "榆林市危险化学品企业动火和受限空间作业安全管理制度",
        "doc_type": "安全管理制度", "discipline": "安全", "doc_no": "榆政办发〔2025〕1号",
        "version": "2025版", "effective_date": "2025-01-09", "status": "有效",
        "classification": "公开", "hazard_tags": ["动火作业", "受限空间作业"],
        "maintenance_type": ["检维修"], "ref_standard": ["GB 30871-2022", "中华人民共和国安全生产法"],
        "source_url": "https://yl.gov.cn/zwgk/zfgb/2025n/dyq/202505/t20250526_2002484.html",
        "publisher": "榆林市人民政府办公室", "note": "第十九条明示原〔2023〕26号废止，可作版本治理正样本",
    },
    "yulin_2023_old": {
        "title": "榆林市危险化学品企业动火和受限空间作业安全管理制度",
        "doc_type": "安全管理制度", "discipline": "安全", "doc_no": "榆政办发〔2023〕26号",
        "version": "2023版", "effective_date": "2023-12-07", "status": "已废止",
        "classification": "公开", "hazard_tags": ["动火作业", "受限空间作业"],
        "maintenance_type": ["检维修"], "ref_standard": ["GB 30871-2022"],
        "source_url": "https://yl.gov.cn/zwgk/zfgb/2023/d5q/202312/t20231207_334122.html",
        "publisher": "榆林市人民政府办公室", "note": "被2025版明示废止；作为'过期版本召回'负样本与冲突检测样本",
    },
    "case_quanzhou_1220": {
        "title": "福建泉州中化学天辰(泉州)新材料有限公司“12·20”一般泄漏爆炸事故调查报告",
        "doc_type": "事故调查报告", "discipline": "安全/工艺", "doc_no": "泉政函〔2024〕55号",
        "version": "2024版", "effective_date": "2024-04-26", "status": "有效",
        "classification": "公开", "hazard_tags": ["泄漏", "爆炸", "压力容器", "超压", "联锁失效"],
        "equipment_tags": ["19-V-4215", "T-4200", "T-4300", "1910-PV-42151", "MP-41009-200-S1B-H1"],
        "maintenance_type": ["故障维修", "检维修"], "ref_standard": ["GB/T 45001", "生产安全事故报告和调查处理条例"],
        "source_url": "https://yjj.quanzhou.gov.cn/zwgk/zfxxgkml/yjgl/202404/t20240426_3031381.htm",
        "publisher": "泉州市人民政府", "note": "含完整位号/设计参数/故障时序，是设备维修轨核心语料",
    },
    "case_quanzhou_rectify_2026": {
        "title": "福建泉州中化学天辰(泉州)新材料有限公司“12·20”事故整改和防范措施落实情况评估报告",
        "doc_type": "整改评估报告", "discipline": "安全", "doc_no": "泉政函〔2024〕55号(评估)",
        "version": "2026版", "effective_date": "2026-05-07", "status": "有效",
        "classification": "公开", "hazard_tags": ["泄漏", "爆炸"],
        "equipment_tags": ["19-V-4215"], "maintenance_type": ["检维修"],
        "ref_standard": ["GB 30871-2022"],
        "source_url": "https://yjj.quanzhou.gov.cn/zwgk/zfxxgkml/yjgl/202605/t20260507_3289165.htm",
        "publisher": "泉州市应急管理局", "note": "设备全生命周期(采购-部署-使用-维护-报废)管理要求",
    },
    "case_gaoming_leak": {
        "title": "茂名高州金展石化有限公司“1·28”一般泄漏事故调查报告",
        "doc_type": "事故调查报告", "discipline": "安全", "doc_no": "",
        "version": "2024版", "effective_date": "2024-10-15", "status": "有效",
        "classification": "公开", "hazard_tags": ["泄漏"], "maintenance_type": [],
        "ref_standard": [],
        "source_url": "https://yjgl.gd.gov.cn/gk/zdlyxxgk/sgdcbg/content/mpost_4506280.html",
        "publisher": "广东省应急管理厅", "note": "正文主体在PDF附件中，HTML为说明页",
    },
    "case_sinopec_maoming_68": {
        "title": "中国石化茂名分公司“6·8”泄漏起火事故调查报告",
        "doc_type": "事故调查报告", "discipline": "安全/设备", "doc_no": "",
        "version": "2023版", "effective_date": "2023-03-18", "status": "有效",
        "classification": "公开", "hazard_tags": ["泄漏", "静电", "带压拆卸"],
        "equipment_tags": ["轨道球阀"], "maintenance_type": ["故障维修", "检维修"], "ref_standard": [],
        "source_url": "https://xt.gov.cn/xt/aqsc/202303/18a458881fce4779a747599bd874ec95.shtml",
        "publisher": "茂名市应急管理局", "note": "违规改造+带压拆卸导致阀杆冲出，设备失效机理典型",
    },
    "plan_tieling_qinghe_tzb": {
        "title": "铁岭市清河区市场监督管理局特种设备突发安全事件应急预案",
        "doc_type": "应急预案", "discipline": "安全/设备", "doc_no": "",
        "version": "2022版", "effective_date": "2022-12-15", "status": "有效",
        "classification": "公开", "hazard_tags": ["锅炉爆炸", "压力容器爆裂", "起重机械倾覆"],
        "equipment_tags": ["起重机械", "有机热载体锅炉"], "maintenance_type": ["故障维修", "应急处置"],
        "ref_standard": ["TSG", "特种设备安全法"],
        "source_url": "http://tlqh.gov.cn/qinghe/zwgk/zfxxgk/fdzdgknr/tfggsj/yjya/2022121515144179398/index.html",
        "publisher": "铁岭市清河区市场监督管理局",
        "note": "含起重机械紧急处置方案、有机热载体锅炉紧急处理（7种紧急停炉情形+停炉步骤）",
    },
    "plan_jiaocheng_tzb": {
        "title": "蕉城区市场监督管理局特种设备事故应急预案",
        "doc_type": "应急预案", "discipline": "安全/设备", "doc_no": "",
        "version": "2023版", "effective_date": "2023-12-13", "status": "有效",
        "classification": "公开", "hazard_tags": ["锅炉爆炸", "压力容器泄漏", "起重机械倾覆"],
        "equipment_tags": ["锅炉", "压力容器", "起重机械"], "maintenance_type": ["应急处置"], "ref_standard": [],
        "source_url": "http://www.jiaocheng.gov.cn/zfxxgkzl/zfxxgkml/zfgbmxxgk/jcqscjdglj/gkml/cpzl/202312/t20231213_1893755.htm",
        "publisher": "宁德市蕉城区市场监督管理局", "note": "危险源因素明确到设备类型",
    },
    "plan_tianjin_dierying": {
        "title": "东二营镇特种设备应急预案",
        "doc_type": "应急预案", "discipline": "安全/设备", "doc_no": "",
        "version": "2025版", "effective_date": "2025-01-03", "status": "有效",
        "classification": "公开", "hazard_tags": ["锅炉爆炸", "压力容器泄漏"],
        "equipment_tags": ["锅炉", "压力容器"], "maintenance_type": ["应急处置"], "ref_standard": [],
        "source_url": "https://www.tjjz.gov.cn/zwgk/zfxxgkxzj/deyz1/fdzdgknr67/zdmsxx67/aqsc67/202501/t20250103_6823885.html",
        "publisher": "天津市蓟州区市场监督管理局", "note": "分设备列应急措施，含119/120/110报告流程",
    },
    "plan_yiyuan_tzb": {
        "title": "沂源县荆山学校特种设备安全应急预案",
        "doc_type": "应急预案", "discipline": "安全/设备", "doc_no": "",
        "version": "2023版", "effective_date": "2023-12-13", "status": "有效",
        "classification": "公开", "hazard_tags": ["特种设备事故"],
        "equipment_tags": ["锅炉", "压力容器", "电梯", "起重机"], "maintenance_type": ["应急处置"],
        "ref_standard": [],
        "source_url": "http://www.yiyuan.gov.cn/gongkai/site_yyxjyhtyj/channel_64b8defbbfe6363923269841/doc_65a0fa512dd32b4326f10abe.html",
        "publisher": "沂源县荆山学校", "note": "规定各设备应急预案应包含的5项内容",
    },
    "plan_haiyang_tzb": {
        "title": "海阳市市场监督管理局特种设备安全事故应急救援预案",
        "doc_type": "应急预案", "discipline": "安全/设备", "doc_no": "",
        "version": "2015版", "effective_date": "2015-07-09", "status": "有效",
        "classification": "公开", "hazard_tags": ["锅炉", "压力容器", "压力管道", "电梯", "起重机械", "厂内机动车辆"],
        "equipment_tags": ["锅炉", "压力容器", "压力管道", "电梯", "起重机械", "厂内机动车辆", "游乐设施"],
        "maintenance_type": ["应急处置"], "ref_standard": [],
        "source_url": "https://www.haiyang.gov.cn/art/2015/7/9/art_49016_1890566.html",
        "publisher": "海阳市市场监督管理局",
        "note": "正文在页面正文容器中仅取到概要；同类内容以 plan_tieling_qinghe_tzb 为主样本",
        "extract_quality": "partial",
    },
    "guide_wuhan_jiayouzhan": {
        "title": "加油站安全常识及应急处置",
        "doc_type": "操作规程/应急处置", "discipline": "安全", "doc_no": "",
        "version": "2020版", "effective_date": "2020-05-14", "status": "有效",
        "classification": "公开", "hazard_tags": ["静电", "火灾", "跑冒油", "雷暴"],
        "equipment_tags": ["加油机", "油罐车", "卸油管", "油气回收装置"],
        "maintenance_type": ["操作", "应急处置"], "ref_standard": ["GB 50156"],
        "source_url": "https://www.dxh.gov.cn/ZWGK/QZFXXGKML/GGFWXX_devsavepath/AQSC/aqsgyj/202005/t20200514_1317560.shtml",
        "publisher": "武汉市东西湖区人民政府", "note": "含静电接地顺序、油罐跑冒油处置步骤等强操作规程内容",
    },
    "guide_jiaxian_guanli_zhidu": {
        "title": "有限空间作业安全生产管理制度",
        "doc_type": "安全管理制度", "discipline": "安全", "doc_no": "",
        "version": "", "effective_date": "", "status": "有效",
        "classification": "公开", "hazard_tags": ["受限空间作业", "中毒窒息"],
        "maintenance_type": ["检维修"], "ref_standard": [],
        "source_url": "https://www.ja.gov.cn/public/6602121/22801321.html",
        "publisher": "嘉祥县人民政府", "note": "制度类，含作业审批与监护职责",
    },
    "guide_shijiu_shoujiu": {
        "title": "有限空间作业事故安全施救指南",
        "doc_type": "应急处置指南", "discipline": "安全", "doc_no": "",
        "version": "2021版", "effective_date": "2021-09-01", "status": "有效",
        "classification": "公开", "hazard_tags": ["受限空间作业", "中毒窒息"],
        "maintenance_type": ["应急处置"], "ref_standard": [],
        "source_url": "https://www.nwserc.cn/yjfw/dzkp/202109/t20210901_397402.shtml",
        "publisher": "国家安全生产应急救援中心", "note": "九项救援要求：个体防护、安全隔离、持续通风、非进入式/进入式救援、轮换救援等",
    },
    "guide_nanchang_fengxian": {
        "title": "有限空间作业安全风险防范指南",
        "doc_type": "操作指南", "discipline": "安全", "doc_no": "",
        "version": "2021版", "effective_date": "2021-06-01", "status": "有效",
        "classification": "公开", "hazard_tags": ["受限空间作业", "中毒窒息"],
        "maintenance_type": ["检维修"], "ref_standard": [],
        "source_url": "https://xhq.nc.gov.cn/xhqrmzf/yjya01/202106/62397e6c61f74d04ad3571d30caf7572.shtml",
        "publisher": "南昌市新建区人民政府", "note": "含'先通风再检测后进入'、机械通风每小时不少于3次、作业前30分钟二次采样",
    },
    "guide_kaiping_xianshi": {
        "title": "受限空间作业安全要点最新指引",
        "doc_type": "操作指引", "discipline": "安全", "doc_no": "",
        "version": "2023版", "effective_date": "2023-11-14", "status": "有效",
        "classification": "公开", "hazard_tags": ["受限空间作业"],
        "maintenance_type": ["检维修"], "ref_standard": [],
        "source_url": "https://www.kaiping.gov.cn/jmkpsmgz/gkmlpt/content/3/3328/post_3328130.html",
        "publisher": "开平市应急管理局", "note": "配套《受限空间安全作业票填写模板》PDF",
    },
    "guide_anhui_xianshi": {
        "title": "有限空间作业安全指南",
        "doc_type": "操作指南", "discipline": "安全", "doc_no": "",
        "version": "2025版", "effective_date": "2025-11-30", "status": "有效",
        "classification": "公开", "hazard_tags": ["受限空间作业", "中毒窒息"],
        "maintenance_type": ["检维修"], "ref_standard": [],
        "source_url": "https://yjt.ah.gov.cn/ztzl/yxkjzyaqzxzz/aqkp/150312111.html",
        "publisher": "安徽省应急管理厅",
        "note": "正文为图片版，网页仅取到摘要；需改用PDF/OCR路线补全",
        "extract_quality": "partial",
    },
    "guide_huaian_xianshi": {
        "title": "有限空间安全作业指南",
        "doc_type": "操作指南", "discipline": "安全", "doc_no": "",
        "version": "2021版", "effective_date": "2021-06-01", "status": "有效",
        "classification": "公开", "hazard_tags": ["受限空间作业", "中毒窒息"],
        "maintenance_type": ["检维修"], "ref_standard": [],
        "source_url": "https://www.huaian.gov.cn/col/13932_321628/art/16645536/1666836301628wAI1FftX.html",
        "publisher": "应急管理部（淮安市转载）",
        "note": "原URL已失效返回专题首页；同源官方PDF已由 mem_xianshi_shouce 覆盖",
        "extract_quality": "failed",
    },
    "mem_xianshi_shouce": {
        "title": "有限空间作业安全指导手册",
        "doc_type": "操作手册", "discipline": "安全", "doc_no": "应急管理部2020年组织编制",
        "version": "2020版", "effective_date": "2020-11-02", "status": "有效",
        "classification": "公开",
        "hazard_tags": ["受限空间作业", "中毒窒息", "缺氧", "燃爆"],
        "maintenance_type": ["检维修"], "ref_standard": ["GBZ 2.1—2019"],
        "source_url": "https://www.mem.gov.cn/gk/tzgg/tz/202010/W020201102332723879957.pdf",
        "publisher": "中华人民共和国应急管理部",
        "note": "★含附录1常见有毒气体浓度判定限值(20种气体mg/m³与ppm双列)、"
               "氧含量判定区间19.5%~23.5%、可燃气体低于爆炸下限10%、"
               "表2-1氧含量对人体影响——数值保真评测的核心素材",
    },
    "guide_shijiazhuang_xianshi": {
        "title": "有限空间作业安全规范（科普版）",
        "doc_type": "操作规范", "discipline": "安全", "doc_no": "",
        "version": "2025版", "effective_date": "2025-10-26", "status": "有效",
        "classification": "公开",
        "hazard_tags": ["受限空间作业", "中毒窒息"], "maintenance_type": ["检维修"],
        "ref_standard": [],
        "source_url": "https://yjglj.sxjz.gov.cn/fmjg/content_354748",
        "publisher": "石家庄市应急管理局", "note": "先检测后作业、检测顺序(氧→可燃→有毒)、每2小时监测、中断超60分钟重测",
    },
    "guide_kepuchina_fanghu": {
        "title": "有限空间作业安全防范措施",
        "doc_type": "操作指南", "discipline": "安全", "doc_no": "",
        "version": "2023版", "effective_date": "2023-05-01", "status": "有效",
        "classification": "公开",
        "hazard_tags": ["受限空间作业", "中毒窒息", "通风"], "maintenance_type": ["检维修"],
        "ref_standard": [],
        "source_url": "https://www.kepuchina.cn/article/articleinfo?business_type=100&ar_id=161113",
        "publisher": "科普中国", "note": "先通风再检测后进入；机械通风每小时不少于3次",
    },
    "guide_liubei_shouce": {
        "title": "有限空间作业安全指导手册（柳北区政府转载）",
        "doc_type": "操作手册", "discipline": "安全", "doc_no": "",
        "version": "2020版", "effective_date": "2020-11-02", "status": "有效",
        "classification": "公开",
        "hazard_tags": ["受限空间作业", "中毒窒息", "缺氧"], "maintenance_type": ["检维修"],
        "ref_standard": ["GBZ 2.1—2019"],
        "source_url": "http://lbq.gov.cn/sjzt/ztzl/dqzt/aqsc1/zsk/t19700101_3292268.shtml",
        "publisher": "柳州市人民政府柳北区人民政府",
        "note": "官方PDF版见 mem_xianshi_shouce；此HTML为转载，可用于多来源一致性校验",
    },
    "mem_2019_78_zhimengdaoze": {
        "title": "危险化学品企业安全风险隐患排查治理导则",
        "doc_type": "排查治理导则", "discipline": "安全", "doc_no": "应急〔2019〕78号",
        "version": "2019版", "effective_date": "2019-08-16", "status": "有效",
        "classification": "公开", "hazard_tags": ["检维修", "作业许可", "承包商管理", "变更管理"],
        "maintenance_type": ["检维修", "点检"], "ref_standard": ["HAZOP", "JHA", "FMEA", "SCL", "LEC", "RAM"],
        "source_url": "https://www.mem.gov.cn/gk/tzgg/tz/201908/W020190816668890436573.pdf",
        "publisher": "中华人民共和国应急管理部",
        "note": "55页含35张表；含巡检频次(两重点一重大≤1h)、14大排查方面，是'表格密集型'主力样本",
    },
    "AQ_T3034_2022_huagong_anquan": {
        "title": "化工过程安全管理导则",
        "doc_type": "行业标准", "discipline": "安全/设备", "doc_no": "AQ/T 3034—2022",
        "version": "2022版", "effective_date": "2023-04-01", "status": "有效",
        "classification": "公开", "hazard_tags": ["设备完好性", "检维修", "作业许可", "承包商管理"],
        "maintenance_type": ["检维修", "预防性维护", "点检"],
        "ref_standard": ["GB/T 1.1—2020"],
        "source_url": "https://www.qss.gov.cn/group3/M00/14/2B/CtG0ZWObzTOAJeK0ABuIHKb5BdU885.pdf",
        "publisher": "中华人民共和国应急管理部（潜山市政府转载）",
        "note": "59页含41张表；4.10设备完好性管理明确要求建立操作规程和检维修规程——设备维修轨的规范依据",
    },
    "gaoqing_yuxian_kongjian": {
        "title": "高青县应急管理局关于开展有限空间作业专项检查的通知",
        "doc_type": "检查通知", "discipline": "安全", "doc_no": "",
        "version": "2023版", "effective_date": "2023-03-13", "status": "有效",
        "classification": "公开", "hazard_tags": ["受限空间作业", "外包管理", "应急准备"],
        "maintenance_type": ["检维修"], "ref_standard": [],
        "source_url": "http://www.gaoqing.gov.cn/resources/public/20230313/640edd5f3f963d17a2350059.pdf",
        "publisher": "高青县应急管理局", "note": "'九到位'检查要点，文本型PDF",
    },
    "kaiping_zuoyepiao": {
        "title": "受限空间安全作业票填写模板",
        "doc_type": "作业票表单", "discipline": "安全", "doc_no": "",
        "version": "2023版", "effective_date": "2023-11-14", "status": "有效",
        "classification": "公开", "hazard_tags": ["受限空间作业"],
        "maintenance_type": ["检维修"], "ref_standard": [],
        "source_url": "http://www.kaiping.gov.cn/jmkpsmgz/attachment/0/346/346167/3328130.pdf",
        "publisher": "开平市应急管理局", "note": "表单型PDF，2页2表；可作QA/结构化表单解析样本",
    },
    "yuncheng_paichabiao": {
        "title": "危险化学品企业安全风险隐患排查表",
        "doc_type": "检查表(表单)", "discipline": "安全", "doc_no": "",
        "version": "2021版", "effective_date": "2021-09-09", "status": "有效",
        "classification": "公开", "hazard_tags": ["隐患排查"],
        "maintenance_type": ["检维修", "点检"], "ref_standard": ["应急〔2019〕78号"],
        "source_url": "https://www.yuncheng.gov.cn/uploadfiles/202109/09/2021090910062421691740.pdf",
        "publisher": "运城市应急管理局",
        "note": "★纯扫描件(无文本层)，用作图片型PDF→OCR链路的真实测试样本",
    },
}

EXT_MD = {"md": "markdown", "html": "html_webpage", "pdf": "pdf_document"}


def main():
    entries = []
    for stem, ann in ANNOTATIONS.items():
        md_path = MD / f"{stem}.md"
        raw_candidates = sorted(RAW.glob(stem + ".*"))
        raw_path = raw_candidates[0] if raw_candidates else None
        ext = raw_path.suffix.lstrip(".") if raw_path else ("md" if md_path.exists() else "")
        text = ""
        if md_path.exists():
            text = md_path.read_text(encoding="utf-8", errors="replace")
        body = re.sub(r"^---.*?---", "", text, flags=re.S)
        entry = {
            "doc_id": stem,
            "title": ann["title"],
            "doc_type": ann["doc_type"],
            "discipline": ann["discipline"],
            "doc_no": ann.get("doc_no", ""),
            "version": ann.get("version", ""),
            "effective_date": ann.get("effective_date", ""),
            "status": ann.get("status", ""),
            "classification": ann.get("classification", "公开"),
            "publisher": ann.get("publisher", ""),
            "source_url": ann["source_url"],
            "source_format": EXT_MD.get(ext, ext),
            "raw_file": raw_path.name if raw_path else None,
            "text_file": md_path.name if md_path.exists() else None,
            "chars": len(body.strip()),
            "hazard_tags": ann.get("hazard_tags", []),
            "equipment_tags": ann.get("equipment_tags", []),
            "maintenance_type": ann.get("maintenance_type", []),
            "ref_standard": ann.get("ref_standard", []),
            "note": ann.get("note", ""),
            "extract_quality": ann.get("extract_quality", "ok"),
        }
        # 自动校正提取质量：字符数过少说明正文没取全；0 字符的 PDF 说明是扫描件
        if entry["extract_quality"] == "ok":
            if entry["chars"] == 0 and entry["source_format"] == "pdf_document":
                entry["extract_quality"] = "ocr_required"
            elif entry["chars"] < 800 and entry["source_format"] != "pdf_document":
                entry["extract_quality"] = "partial"
        entries.append(entry)

    out_dir = ROOT / "data"

    # 🔴 版本关系字段由 src/versioning.py 补齐——**必须在这里接上**，
    #    否则重跑本脚本会把 effective_from / effective_to / superseded_by 等
    #    6 个字段冲掉（它们不是人工标注的，是算出来的）。
    #    顺序：先按人工标注生成，再由版本模块补算。
    try:
        from src.versioning import apply_version_fields
        entries, conflicts = apply_version_fields(entries, out_dir / "md")
        for c in conflicts:
            print(f"  [版本冲突] {c['doc_id']}: 人工标注 {c['recorded']} "
                  f"→ 正文明示 {c['declared']}（{c['resolution']}）")
    except ImportError as e:
        # 🔴 **直接失败，不写 manifest**。版本字段是算出来的、不是人工标的，
        # 跳过这一步就等于把 M1 的成果静默冲掉 —— 而 README 让人直接跑本脚本。
        raise SystemExit(
            "未能导入 src.versioning，manifest 未写入（避免冲掉 M1 的版本字段）。\n"
            f"  原因：{e}\n"
            "  修法：确认 src/versioning.py 存在且可导入；"
            "或改跑 python tools/build_version_fields.py 单独补字段。") from e

    (out_dir / "manifest.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    # 可读清单
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    lines = ["# 语料清单 · 工业设备维修与安全规程问答 RAG", "",
             f"生成时间：{__import__('datetime').date.today()} | 文档总数：{len(entries)}", "",
             "全部文档均为**中国政府网站公开文件**（应急管理部 / 省级应急厅 / 市政府 / 市场监督管理局），",
             "可公开引用；仅供内部 RAG 测试与评测使用，不得对外再分发。", "",
             "| doc_id | 标题 | 类型 | 专业 | 文号 | 版本/生效 | 状态 | 格式 | 字符数 | 提取质量 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for e in sorted(entries, key=lambda x: (x["discipline"], x["doc_type"])):
        q = {"ok": "完整", "partial": "部分", "failed": "失败", "ocr_required": "需OCR"}.get(
            e["extract_quality"], e["extract_quality"])
        lines.append(
            f"| `{e['doc_id']}` | {e['title'][:38]} | {e['doc_type']} | {e['discipline']} | "
            f"{e['doc_no'] or '-'} | {e['version'] or '-'} / {e['effective_date'] or '-'} | "
            f"{e['status']} | {e['source_format']} | {e['chars']} | {q} |")
    lines += ["", "## 危险源标签分布", "", "| 危险源 | 文档数 |", "|---|---|"]
    from collections import Counter
    c = Counter(h for e in entries for h in e["hazard_tags"])
    for k, v in c.most_common():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## 文档类型分布", "", "| 类型 | 数量 |", "|---|---|"]
    for k, v in Counter(e["doc_type"] for e in entries).most_common():
        lines.append(f"| {k} | {v} |")
    (docs_dir / "语料清单.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"manifest -> {out_dir / 'manifest.json'}")
    print(f"清单     -> {docs_dir / '语料清单.md'}")
    print(f"文档数   {len(entries)} | 总字符 {sum(e['chars'] for e in entries)}")


if __name__ == "__main__":
    main()
