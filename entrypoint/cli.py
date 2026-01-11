"""
CCBA RAG System - Command Line Interface

Usage:
    ccba query "Your question here"
    ccba index data/
    ccba status
"""

import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(
    name="ccba",
    help="CCBA RAG System - Vietnamese Construction Standards Expert",
    add_completion=False
)
console = Console()


def get_system():
    """Lazy load RAGSystem to avoid import costs on every CLI call."""
    from ccba_rag.core.rag_system import RAGSystem
    return RAGSystem(verbose=False)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask the RAG system"),
    top_k: int = typer.Option(100, "--top-k", "-k", help="Number of retrieval candidates"),
    top_n: int = typer.Option(5, "--top-n", "-n", help="Number of final results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    provider: str = typer.Option("gemini", "--provider", "-p", help="LLM provider: gemini, groq, deepseek"),
):
    """
    Query the RAG system with a question.
    
    Example:
        ccba query "Chiều cao tối thiểu của tầng 1 là bao nhiêu?"
    """
    with console.status("[bold green]Initializing RAG System..."):
        system = get_system()
    
    with console.status("[bold blue]Processing query..."):
        result = system.query(
            question=question,
            verbose=False,
            top_k=top_k,
            top_n=top_n
        )
    
    # Display answer
    console.print(Panel(
        result.get('answer', 'No answer generated'),
        title="[bold green]Answer",
        border_style="green"
    ))
    
    # Display statistics
    stats = result.get('stats', {})
    model = result.get('model', 'Unknown')
    used_fallback = result.get('used_fallback', False)
    
    stats_table = Table(show_header=False, box=None)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="yellow")
    
    stats_table.add_row("Model", model + (" (fallback)" if used_fallback else ""))
    stats_table.add_row("Retrieval", f"{stats.get('retrieval_ms', 0):.0f}ms")
    stats_table.add_row("Reranking", f"{stats.get('reranking_ms', 0):.0f}ms")
    stats_table.add_row("Generation", f"{stats.get('generation_ms', 0):.0f}ms")
    stats_table.add_row("Total", f"{stats.get('total_ms', 0):.0f}ms")
    stats_table.add_row("Contexts", str(stats.get('final_count', 0)))
    
    console.print(Panel(stats_table, title="[bold blue]Statistics", border_style="blue"))
    
    # Show contexts if verbose
    if verbose:
        contexts = result.get('contexts', [])
        for i, ctx in enumerate(contexts[:3], 1):
            console.print(f"\n[dim]Context {i}: {ctx.get('document_name', 'Unknown')}[/dim]")
            console.print(f"[dim]{ctx.get('text', '')[:200]}...[/dim]")


@app.command()
def index(
    directory: str = typer.Argument("data", help="Directory containing documents"),
    drop_existing: bool = typer.Option(False, "--drop", "-d", help="Drop existing collection first"),
):
    """
    Index documents from a directory.
    
    Example:
        ccba index data/pdfs/
        ccba index data/ --drop
    """
    path = Path(directory)
    if not path.exists():
        console.print(f"[red]Error: Directory '{directory}' does not exist[/red]")
        raise typer.Exit(1)
    
    with console.status(f"[bold green]Indexing documents from {directory}..."):
        system = get_system()
        system.index_documents(str(path), drop_existing=drop_existing)
    
    console.print(f"[green]✓ Indexing complete![/green]")


@app.command()
def status():
    """
    Show system status and configuration.
    """
    from ccba_rag.core.settings import settings
    
    table = Table(title="CCBA RAG System Status")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Milvus Host", settings.milvus_host)
    table.add_row("Collection", settings.milvus_collection_name)
    table.add_row("Embedding Model", settings.bge_model_name)
    table.add_row("Max Length", str(settings.bge_max_length))
    table.add_row("Reranker Enabled", str(settings.enable_reranker))
    table.add_row("Primary LLM", settings.gemini_model)
    table.add_row("Fallback LLM", settings.groq_model)
    
    console.print(table)
    
    # Check collection
    try:
        from ccba_rag.retrieval.vectorstores.milvus import MilvusStore
        store = MilvusStore()
        if store.has_collection():
            console.print(f"\n[green]✓ Collection '{settings.milvus_collection_name}' exists[/green]")
        else:
            console.print(f"\n[yellow]⚠ Collection '{settings.milvus_collection_name}' does not exist[/yellow]")
    except Exception as e:
        console.print(f"\n[red]✗ Cannot connect to Milvus: {e}[/red]")


@app.command()
def retrieve(
    question: str = typer.Argument(..., help="Question to retrieve contexts for"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
):
    """
    Retrieve contexts without generating an answer.
    
    Example:
        ccba retrieve "phòng cháy chữa cháy" --top-k 5
    """
    with console.status("[bold green]Initializing..."):
        system = get_system()
    
    with console.status("[bold blue]Retrieving..."):
        result = system.retrieve(question, top_k=top_k, top_n=top_k)
    
    contexts = result.get('contexts', [])
    
    console.print(f"\n[bold]Found {len(contexts)} contexts:[/bold]\n")
    
    for i, ctx in enumerate(contexts, 1):
        doc_name = ctx.get('document_name', 'Unknown')
        score = ctx.get('rerank_score', ctx.get('retrieval_score', 0))
        text = ctx.get('text', '')[:300]
        
        console.print(f"[bold cyan]{i}. {doc_name}[/bold cyan] (score: {score:.3f})")
        console.print(f"   {text}...")
        console.print()


if __name__ == "__main__":
    app()
