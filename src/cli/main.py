import click
import asyncio
import json
import yaml
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

from ..llm.model_router import ModelRouter
from ..prompts.manager import PromptManager
from ..crawler.arxiv_crawler import ArxivCrawler
from ..analyzer.requirement_analyzer import RequirementAnalyzer
from ..analyzer.classifier import PaperClassifier
from ..reviewer.generator import ReviewGenerator
from ..core.models import SearchQuery, TaskType
from ..utils.config_utils import setup_proxy_config
from ..utils.llm_logger import get_llm_logger

console = Console()

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """AI文献自动爬取助手 - 智能文献调研工具"""
    pass

@cli.command()
@click.argument('query')
@click.option('--max-papers', default=20, help='最大爬取论文数量')
@click.option('--model', help='指定使用的模型 (格式: provider_name/model_name, 如: openai_proxy/gpt-4)')
@click.option('--output', default='./papers.json', help='输出文件路径')
@click.option('--config', default='config/config.yaml', help='配置文件路径')
def search(query, max_papers, model, output, config):
    """搜索并分析文献"""
    asyncio.run(_search_papers(query, max_papers, model, output, config))

async def _search_papers(query: str, max_papers: int, model: str, output: str, config_path: str):
    """搜索论文的异步实现"""
    try:
        console.print(Panel(f"[bold blue]开始文献搜索：{query}[/bold blue]", expand=False))
        
        # 初始化组件
        model_router, prompt_manager = await _initialize_components(config_path)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # 步骤1：分析用户需求
            task1 = progress.add_task("分析研究需求...", total=None)
            analyzer = RequirementAnalyzer(model_router, prompt_manager)
            requirement = await analyzer.analyze_requirement(query, model)
            progress.update(task1, description="需求分析完成")
            
            # 显示分析结果
            _display_requirement_analysis(requirement)
            
            # 步骤2：爬取论文
            task2 = progress.add_task("爬取ArXiv论文...", total=None)
            async with ArxivCrawler() as crawler:
                # 使用分析得到的搜索查询
                search_queries = requirement.search_queries[:3]  # 使用前3个查询
                all_papers = []
                
                for search_query in search_queries:
                    query_obj = SearchQuery(
                        query_string=search_query,
                        max_results=max_papers // len(search_queries)
                    )
                    result = await crawler.fetch_papers(query_obj)
                    all_papers.extend(result.papers)
                
                # 去重
                unique_papers = crawler.remove_duplicates(all_papers)
                papers = unique_papers[:max_papers]
            
            progress.update(task2, description=f"爬取完成 ({len(papers)}篇论文)")
            
            # 记录爬取到的ArXiv链接
            llm_logger = get_llm_logger()
            llm_logger.log_arxiv_links(papers)
            
            # 步骤3：分类论文
            task3 = progress.add_task("分类和分析论文...", total=None)
            classifier = PaperClassifier(model_router, prompt_manager)
            classifications = await classifier.classify_papers(papers, model)
            progress.update(task3, description="论文分类完成")
        
        # 显示结果
        _display_search_results(papers, classifications)
        
        # 保存结果
        _save_results(papers, classifications, requirement, output)
        
        console.print(f"\n[green]搜索完成！结果已保存到 {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]搜索失败: {str(e)}[/red]")

@cli.command()
@click.option('--input', default='./papers.json', help='论文数据文件路径')
@click.option('--output', default='./review.md', help='综述输出文件路径')
@click.option('--model', help='指定使用的模型')
@click.option('--config', default='config/config.yaml', help='配置文件路径')
def review(input_file, output, model, config):
    """生成文献综述"""
    asyncio.run(_generate_review(input_file, output, model, config))

async def _generate_review(input_file: str, output: str, model: str, config_path: str):
    """生成综述的异步实现"""
    try:
        console.print(Panel("[bold blue]开始生成文献综述[/bold blue]", expand=False))
        
        # 加载论文数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = [paper for paper in data['papers']]
        classifications = [cls for cls in data['classifications']]
        requirement = data['requirement']
        
        # 初始化组件
        model_router, prompt_manager = await _initialize_components(config_path)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("生成综述中...", total=None)
            
            # 生成综述
            generator = ReviewGenerator(model_router, prompt_manager)
            
            # 重建对象（简化处理）
            from ..core.models import Paper, PaperClassification, ResearchRequirement
            paper_objects = [Paper(**p) for p in papers]
            classification_objects = [PaperClassification(**c) for c in classifications]
            requirement_object = ResearchRequirement(**requirement)
            
            review = await generator.generate_literature_review(
                paper_objects, classification_objects, requirement_object, model
            )
            
            progress.update(task, description="综述生成完成")
        
        # 保存综述
        markdown_content = generator.export_to_markdown(review)
        with open(output, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        console.print(f"\n[green]综述已生成并保存到 {output}[/green]")
        console.print(f"📊 综述统计: {review.word_count} 字")
        
    except Exception as e:
        console.print(f"[red]综述生成失败: {str(e)}[/red]")

@cli.command()
@click.option('--config', default='config/config.yaml', help='配置文件路径')
def test_models(config):
    """测试模型连接"""
    asyncio.run(_test_models(config))

async def _test_models(config_path: str):
    """测试模型连接的异步实现"""
    try:
        model_router, _ = await _initialize_components(config_path)
        
        console.print(Panel("[bold blue]测试模型连接[/bold blue]", expand=False))
        
        results = model_router.test_all_connections()
        
        table = Table(title="模型连接状态")
        table.add_column("模型", style="cyan")
        table.add_column("状态", style="magenta")
        
        for model_name, is_connected in results.items():
            status = "连接正常" if is_connected else "连接失败"
            table.add_row(model_name, status)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]测试失败: {str(e)}[/red]")

@cli.command()
@click.option('--config', default='config/config.yaml', help='配置文件路径')
def stats(config):
    """显示使用统计"""
    asyncio.run(_show_stats(config))

@cli.command()
@click.option('--api-key', required=True, help='代理服务的API密钥')
@click.option('--base-url', required=True, help='代理服务的base_url')
@click.option('--config', default='config/config.yaml', help='配置文件路径')
def setup_proxy(api_key, base_url, config):
    """快速配置第三方代理"""
    _setup_proxy_config(api_key, base_url, config)

@cli.command()
@click.option('--config', default='config/config.yaml', help='配置文件路径')
def list_models(config):
    """列出所有可用的模型"""
    asyncio.run(_list_models(config))

async def _show_stats(config_path: str):
    """显示统计的异步实现"""
    try:
        model_router, _ = await _initialize_components(config_path)
        
        console.print(Panel("[bold blue]使用统计[/bold blue]", expand=False))
        
        stats_data = model_router.get_all_stats()
        
        table = Table(title="模型使用统计")
        table.add_column("模型", style="cyan")
        table.add_column("请求数", style="magenta")
        table.add_column("成功率", style="green")
        table.add_column("总成本", style="yellow")
        
        for model_name, stats in stats_data.items():
            success_rate = f"{stats['success_rate']:.2%}"
            total_cost = f"${stats['total_cost']:.4f}"
            table.add_row(
                model_name, 
                str(stats['request_count']), 
                success_rate, 
                total_cost
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]❌ 统计显示失败: {str(e)}[/red]")

async def _initialize_components(config_path: str):
    """初始化核心组件"""
    # 初始化模型路由器
    model_router = ModelRouter()
    
    # 加载默认配置（如果配置文件存在）
    if Path(config_path).exists():
        model_router.load_config(config_path)
    else:
        # 创建默认配置
        _create_default_config(config_path)
    
    # 初始化提示词管理器
    prompt_manager = PromptManager()
    
    return model_router, prompt_manager

def _create_default_config(config_path: str):
    """创建默认配置文件"""
    default_config = {
        "models": {
            "openai_gpt4": {
                "provider": "openai",
                "model_name": "gpt-4",
                "api_key": "${OPENAI_API_KEY}",
                "temperature": 0.7,
                "max_tokens": 4000
            }
        },
        "task_preferences": {
            "requirement_analysis": ["openai_gpt4"],
            "paper_classification": ["openai_gpt4"],
            "review_generation": ["openai_gpt4"]
        }
    }
    
    # 确保目录存在
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, allow_unicode=True, default_flow_style=False)

def _display_requirement_analysis(requirement):
    """显示需求分析结果"""
    console.print("\n[bold yellow]需求分析结果:[/bold yellow]")
    
    table = Table(show_header=False, box=None)
    table.add_column("项目", style="cyan", width=15)
    table.add_column("内容", style="white")
    
    table.add_row("研究领域", ", ".join(requirement.research_domains))
    table.add_row("具体主题", ", ".join(requirement.specific_topics))
    table.add_row("关键词", ", ".join(requirement.keywords))
    table.add_row("预期论文数", str(requirement.paper_count_estimate))
    
    console.print(table)

def _display_search_results(papers, classifications):
    """显示搜索结果"""
    console.print(f"\n[bold yellow]📚 找到 {len(papers)} 篇相关论文:[/bold yellow]")
    
    # 创建分类统计
    from collections import defaultdict
    domain_counts = defaultdict(int)
    for cls in classifications:
        domain_counts[cls.primary_domain] += 1
    
    # 显示领域分布
    table = Table(title="研究领域分布")
    table.add_column("领域", style="cyan")
    table.add_column("论文数量", style="magenta")
    
    for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
        table.add_row(domain, str(count))
    
    console.print(table)
    
    # 显示高质量论文
    high_quality_papers = [
        (papers[i], classifications[i]) 
        for i in range(len(papers)) 
        if i < len(classifications) and classifications[i].relevance_score > 7.0
    ]
    
    if high_quality_papers:
        console.print(f"\n[bold green]⭐ 高质量论文 (评分>7.0):[/bold green]")
        for i, (paper, cls) in enumerate(high_quality_papers[:5]):
            console.print(f"{i+1}. [bold]{paper.title}[/bold]")
            console.print(f"   评分: {cls.relevance_score:.1f} | 领域: {cls.primary_domain}")
            console.print(f"   [link]{paper.url}[/link]\n")

def _save_results(papers, classifications, requirement, output_path: str):
    """保存搜索结果"""
    # 转换为可序列化的格式
    papers_data = []
    for paper in papers:
        papers_data.append({
            'title': paper.title,
            'authors': paper.authors,
            'abstract': paper.abstract,
            'arxiv_id': paper.arxiv_id,
            'url': paper.url,
            'pdf_url': paper.pdf_url,
            'published_date': paper.published_date,
            'categories': paper.categories
        })
    
    classifications_data = []
    for cls in classifications:
        classifications_data.append({
            'paper_id': cls.paper_id,
            'primary_domain': cls.primary_domain,
            'sub_domains': cls.sub_domains,
            'research_type': cls.research_type.value,
            'technical_approaches': cls.technical_approaches,
            'relevance_score': cls.relevance_score,
            'keywords_extracted': cls.keywords_extracted,
            'application_areas': cls.application_areas,
            'novelty_level': cls.novelty_level.value,
            'citation_potential': cls.citation_potential
        })
    
    requirement_data = {
        'user_input': requirement.user_input,
        'research_domains': requirement.research_domains,
        'specific_topics': requirement.specific_topics,
        'keywords': requirement.keywords,
        'time_preference': requirement.time_preference,
        'paper_count_estimate': requirement.paper_count_estimate,
        'research_focus': requirement.research_focus,
        'search_queries': requirement.search_queries
    }
    
    result_data = {
        'papers': papers_data,
        'classifications': classifications_data,
        'requirement': requirement_data,
        'summary': {
            'total_papers': len(papers),
            'domains_found': len(set(cls.primary_domain for cls in classifications)),
            'avg_relevance_score': sum(cls.relevance_score for cls in classifications) / len(classifications) if classifications else 0
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

def _setup_proxy_config(api_key: str, base_url: str, config_path: str):
    """配置代理设置"""
    try:
        console.print(Panel("[bold blue]配置第三方代理[/bold blue]", expand=False))
        
        # 验证base_url格式
        if not (base_url.startswith("http://") or base_url.startswith("https://")):
            console.print("[red]❌ base_url 必须以 http:// 或 https:// 开头[/red]")
            return
        
        # 添加配置到.env文件
        result = setup_proxy_config(api_key, base_url)
        console.print(f"[green]✅ {result}[/green]")
        
        # 显示使用说明
        console.print("\n[yellow]使用说明:[/yellow]")
        console.print("1. 现在可以使用代理模型了（如 openai_gpt4_proxy）")
        console.print("2. 代理模型在任务偏好中拥有更高优先级")
        console.print("3. 测试连接: python main.py test-models")
        console.print("4. 开始搜索: python main.py search \"你的研究问题\"")
        
        # 测试代理连接
        console.print("\n[cyan]测试代理连接...[/cyan]")
        try:
            model_router = ModelRouter()
            if Path(config_path).exists():
                model_router.load_config(config_path)
            
            # 测试代理模型
            proxy_models = [name for name in model_router.providers.keys() if 'proxy' in name]
            if proxy_models:
                for model_name in proxy_models[:1]:  # 只测试第一个代理模型
                    is_working = model_router.providers[model_name].test_connection()
                    status = "✅ 连接正常" if is_working else "❌ 连接失败"
                    console.print(f"  {model_name}: {status}")
            else:
                console.print("[yellow]  没有找到代理模型配置[/yellow]")
                
        except Exception as e:
            console.print(f"[yellow]  连接测试失败: {str(e)}[/yellow]")
        
    except Exception as e:
        console.print(f"[red]❌ 配置失败: {str(e)}[/red]")

async def _list_models(config_path: str):
    """列出可用模型"""
    try:
        console.print(Panel("[bold blue]可用模型列表[/bold blue]", expand=False))
        
        model_router = ModelRouter()
        if Path(config_path).exists():
            model_router.load_config(config_path)
        
        available_models = model_router.get_available_models()
        
        if not available_models:
            console.print("[yellow]没有找到可用的模型，请检查配置和API密钥[/yellow]")
            return
        
        for provider_name, models in available_models.items():
            console.print(f"\n[bold cyan]{provider_name}[/bold cyan]")
            
            provider_config = model_router.app_config.providers.get(provider_name)
            if provider_config:
                console.print(f"   类型: {provider_config.provider_type}")
                if provider_config.base_url:
                    console.print(f"   地址: {provider_config.base_url}")
                console.print(f"   模型数量: {len(models)}")
                
                # 显示每个模型
                for model_name in models:
                    model_ref = f"{provider_name}/{model_name}"
                    model_info = model_router.get_model_info(model_ref)
                    
                    if model_info:
                        cost_info = f"${model_info.cost_per_1k_input:.3f}/${model_info.cost_per_1k_output:.3f}"
                        console.print(f"     {model_info.display_name} ({model_name}) - {cost_info}/1k tokens")
                    else:
                        console.print(f"     {model_name}")
        
        # 显示任务偏好
        console.print(f"\n[bold green]任务偏好配置[/bold green]")
        for task_name, preferred_models in model_router.app_config.task_preferences.items():
            console.print(f"   {task_name}:")
            for i, model_ref in enumerate(preferred_models[:3]):  # 只显示前3个
                console.print(f"     {i+1}. {model_ref}")
            if len(preferred_models) > 3:
                console.print(f"     ... (+{len(preferred_models)-3} 更多)")
        
        # 使用示例
        console.print(f"\n[bold yellow]使用示例[/bold yellow]")
        if available_models:
            first_provider = list(available_models.keys())[0]
            first_model = available_models[first_provider][0]
            example_ref = f"{first_provider}/{first_model}"
            console.print(f"   python main.py search \"研究问题\" --model {example_ref}")
        
    except Exception as e:
        console.print(f"[red]❌ 获取模型列表失败: {str(e)}[/red]")

if __name__ == '__main__':
    cli()