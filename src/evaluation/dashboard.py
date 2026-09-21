"""
src/evaluation/dashboard.py

Evaluation Dashboard Generator

Interactive HTML dashboard oluşturma sistemi:
- Metrics Visualization: Charts ve graphs (Chart.js ile)
- Benchmark Results: Interactive tables
- Model Leaderboard: Ranking ve comparison
- Quality Tracking: Regression alerts ve trend görselleştirme
- Standalone HTML: Server gerektirmeden çalışan dashboard

Bu modül standalone HTML dashboard generate eder:
- Self-contained: Tek HTML dosyası, dependencies CDN'den
- Interactive: JavaScript ile interactive charts
- Responsive: Mobile-friendly design
- Export-ready: Print ve screenshot friendly
- Production-ready: Comprehensive evaluation reports

Dashboard Components:
    Overview Section:
        - Model summary
        - Key metrics
        - Overall scores
    
    Metrics Charts:
        - Line charts for trends
        - Bar charts for comparisons
        - Radar charts for multi-metric view
    
    Benchmark Results:
        - Sortable tables
        - Color-coded performance
        - Expandable details
    
    Quality Tracking:
        - Alert history
        - Regression timeline
        - Trend indicators

Kaynaklar:
    - Chart.js for visualization
    - Bootstrap for styling
    - https://www.chartjs.org/

Usage:
    >>> from src.evaluation.dashboard import DashboardGenerator
    >>> 
    >>> # Create generator
    >>> dashboard = DashboardGenerator(title="Model Evaluation")
    >>> 
    >>> # Add results
    >>> dashboard.add_evaluation_results(model_name, results)
    >>> dashboard.add_comparison_report(comparison_report)
    >>> dashboard.add_quality_tracking(quality_tracker)
    >>> 
    >>> # Generate HTML
    >>> dashboard.generate("evaluation_dashboard.html")
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.metrics import EvaluationResults
from src.evaluation.comparison import ComparisonReport
from src.evaluation.quality_tracker import QualityTracker

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================
# Dashboard Generator
# =========================

class DashboardGenerator:
    """
    HTML dashboard generator.
    
    Generates standalone interactive HTML dashboards.
    """
    
    def __init__(
        self,
        title: str = "Evaluation Dashboard",
        description: Optional[str] = None
    ):
        """
        Initialize dashboard generator.
        
        Args:
            title: Dashboard title
            description: Dashboard description
        """
        self.title = title
        self.description = description or "Model Evaluation Results"
        
        self.evaluations: Dict[str, EvaluationResults] = {}
        self.comparison_report: Optional[ComparisonReport] = None
        self.quality_tracker: Optional[QualityTracker] = None
        
        logger.info(f"DashboardGenerator initialized: {title}")
    
    def add_evaluation_results(
        self,
        model_name: str,
        results: EvaluationResults
    ) -> None:
        """
        Add evaluation results for a model.
        
        Args:
            model_name: Model name
            results: Evaluation results
        """
        self.evaluations[model_name] = results
        logger.info(f"Added evaluation results for: {model_name}")
    
    def add_comparison_report(
        self,
        report: ComparisonReport
    ) -> None:
        """
        Add comparison report.
        
        Args:
            report: Comparison report
        """
        self.comparison_report = report
        logger.info("Added comparison report")
    
    def add_quality_tracking(
        self,
        tracker: QualityTracker
    ) -> None:
        """
        Add quality tracker.
        
        Args:
            tracker: Quality tracker
        """
        self.quality_tracker = tracker
        logger.info("Added quality tracker")
    
    def generate(self, output_path: str) -> None:
        """
        Generate HTML dashboard.
        
        Args:
            output_path: Output HTML file path
        """
        logger.info(f"Generating dashboard: {output_path}")
        
        html = self._generate_html()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"✓ Dashboard generated: {output_path}")
    
    def _generate_html(self) -> str:
        """Generate complete HTML."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            padding: 20px;
        }}
        
        .dashboard-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .card {{
            border: none;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border-radius: 10px;
        }}
        
        .card-header {{
            background-color: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
            font-weight: 600;
        }}
        
        .metric-card {{
            text-align: center;
            padding: 20px;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .metric-label {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        
        .table-hover tbody tr:hover {{
            background-color: #f8f9fa;
        }}
        
        .badge-winner {{
            background-color: #28a745;
        }}
        
        .alert-critical {{
            border-left: 4px solid #dc3545;
        }}
        
        .alert-error {{
            border-left: 4px solid #fd7e14;
        }}
        
        .alert-warning {{
            border-left: 4px solid #ffc107;
        }}
        
        .alert-info {{
            border-left: 4px solid #17a2b8;
        }}
        
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 20px 0;
        }}
        
        .footer {{
            margin-top: 40px;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
        }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <!-- Header -->
        <div class="dashboard-header">
            <h1>{self.title}</h1>
            <p>{self.description}</p>
            <small>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</small>
        </div>
        
        <!-- Overview Section -->
        {self._generate_overview_section()}
        
        <!-- Metrics Section -->
        {self._generate_metrics_section()}
        
        <!-- Comparison Section -->
        {self._generate_comparison_section()}
        
        <!-- Quality Tracking Section -->
        {self._generate_quality_section()}
        
        <!-- Footer -->
        <div class="footer">
            <p>AI Research Lab - Evaluation Dashboard</p>
            <p><small>Powered by Chart.js & Bootstrap</small></p>
        </div>
    </div>
    
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- Charts -->
    <script>
        {self._generate_chart_scripts()}
    </script>
</body>
</html>"""
    
    def _generate_overview_section(self) -> str:
        """Generate overview section."""
        if not self.evaluations:
            return ""
        
        html = ['<div class="row mb-4">']
        html.append('<div class="col-12"><h2>Overview</h2></div>')
        
        # Key metrics cards
        for model_name, results in self.evaluations.items():
            html.append(f'<div class="col-md-4">')
            html.append('<div class="card metric-card">')
            html.append(f'<h5>{model_name}</h5>')
            html.append(f'<div class="metric-value">{len(results.metrics)}</div>')
            html.append('<div class="metric-label">Metrics Evaluated</div>')
            html.append(f'<div class="mt-2"><small>{results.num_samples:,} samples, {results.num_tokens:,} tokens</small></div>')
            html.append('</div></div>')
        
        html.append('</div>')
        return '\n'.join(html)
    
    def _generate_metrics_section(self) -> str:
        """Generate metrics section."""
        if not self.evaluations:
            return ""
        
        html = ['<div class="row mb-4">']
        html.append('<div class="col-12"><h2>Model Metrics</h2></div>')
        
        for model_name, results in self.evaluations.items():
            html.append('<div class="col-12 mb-3">')
            html.append('<div class="card">')
            html.append(f'<div class="card-header">{model_name}</div>')
            html.append('<div class="card-body">')
            html.append('<table class="table table-hover">')
            html.append('<thead><tr><th>Metric</th><th>Value</th><th>Unit</th><th>Direction</th></tr></thead>')
            html.append('<tbody>')
            
            for metric in results.metrics:
                direction = "↑" if metric.higher_is_better else "↓"
                html.append(f'<tr>')
                html.append(f'<td><strong>{metric.name}</strong></td>')
                html.append(f'<td>{metric.value:.4f}</td>')
                html.append(f'<td>{metric.unit}</td>')
                html.append(f'<td>{direction}</td>')
                html.append(f'</tr>')
            
            html.append('</tbody></table>')
            html.append('</div></div></div>')
        
        html.append('</div>')
        return '\n'.join(html)
    
    def _generate_comparison_section(self) -> str:
        """Generate comparison section."""
        if not self.comparison_report:
            return ""
        
        html = ['<div class="row mb-4">']
        html.append('<div class="col-12"><h2>Model Comparison</h2></div>')
        
        # Overall winner
        html.append('<div class="col-12 mb-3">')
        html.append('<div class="card">')
        html.append('<div class="card-header">Overall Winner</div>')
        html.append('<div class="card-body text-center">')
        html.append(f'<h3><span class="badge badge-winner">{self.comparison_report.overall_winner}</span></h3>')
        html.append('</div></div></div>')
        
        # Metric-wise comparison
        html.append('<div class="col-12 mb-3">')
        html.append('<div class="card">')
        html.append('<div class="card-header">Metric-wise Winners</div>')
        html.append('<div class="card-body">')
        html.append('<table class="table table-hover">')
        html.append('<thead><tr><th>Metric</th><th>Winner</th><th>Mean Scores</th></tr></thead>')
        html.append('<tbody>')
        
        for metric_name, comparison in self.comparison_report.metric_comparisons.items():
            mean_scores_str = ", ".join(
                f"{model}: {comparison.model_scores[model][0]:.4f}"
                for model in comparison.model_scores.keys()
            )
            html.append(f'<tr>')
            html.append(f'<td><strong>{metric_name}</strong></td>')
            html.append(f'<td><span class="badge bg-success">{comparison.winner}</span></td>')
            html.append(f'<td><small>{mean_scores_str}</small></td>')
            html.append(f'</tr>')
        
        html.append('</tbody></table>')
        html.append('</div></div></div>')
        
        # Comparison chart
        html.append('<div class="col-12 mb-3">')
        html.append('<div class="card">')
        html.append('<div class="card-header">Performance Comparison</div>')
        html.append('<div class="card-body">')
        html.append('<div class="chart-container">')
        html.append('<canvas id="comparisonChart"></canvas>')
        html.append('</div></div></div></div>')
        
        html.append('</div>')
        return '\n'.join(html)
    
    def _generate_quality_section(self) -> str:
        """Generate quality tracking section."""
        if not self.quality_tracker:
            return ""
        
        summary = self.quality_tracker.get_summary()
        
        html = ['<div class="row mb-4">']
        html.append('<div class="col-12"><h2>Quality Tracking</h2></div>')
        
        # Summary stats
        html.append('<div class="col-md-4">')
        html.append('<div class="card metric-card">')
        html.append(f'<div class="metric-value">{summary["num_evaluations"]}</div>')
        html.append('<div class="metric-label">Evaluations</div>')
        html.append('</div></div>')
        
        html.append('<div class="col-md-4">')
        html.append('<div class="card metric-card">')
        html.append(f'<div class="metric-value">{summary["num_alerts"]}</div>')
        html.append('<div class="metric-label">Total Alerts</div>')
        html.append('</div></div>')
        
        html.append('<div class="col-md-4">')
        html.append('<div class="card metric-card">')
        critical_count = summary["alert_breakdown"].get("critical", 0)
        html.append(f'<div class="metric-value text-danger">{critical_count}</div>')
        html.append('<div class="metric-label">Critical Alerts</div>')
        html.append('</div></div>')
        
        # Alerts
        if self.quality_tracker.alerts:
            html.append('<div class="col-12 mt-3">')
            html.append('<div class="card">')
            html.append('<div class="card-header">Recent Alerts</div>')
            html.append('<div class="card-body">')
            
            for alert in self.quality_tracker.alerts[-10:]:  # Last 10
                alert_class = f"alert-{alert.level.value}"
                html.append(f'<div class="alert {alert_class}">')
                html.append(f'<strong>[{alert.level.value.upper()}]</strong> {alert.metric_name}: {alert.message}<br>')
                html.append(f'<small>Current: {alert.current_value:.4f}, Best: {alert.best_value:.4f}, Degradation: {alert.degradation_pct:.2f}%</small>')
                html.append('</div>')
            
            html.append('</div></div></div>')
        
        # Quality trends chart
        html.append('<div class="col-12 mt-3">')
        html.append('<div class="card">')
        html.append('<div class="card-header">Quality Trends</div>')
        html.append('<div class="card-body">')
        html.append('<div class="chart-container">')
        html.append('<canvas id="qualityChart"></canvas>')
        html.append('</div></div></div></div>')
        
        html.append('</div>')
        return '\n'.join(html)
    
    def _generate_chart_scripts(self) -> str:
        """Generate Chart.js scripts."""
        scripts = []
        
        # Comparison chart
        if self.comparison_report:
            scripts.append(self._generate_comparison_chart_script())
        
        # Quality trends chart
        if self.quality_tracker:
            scripts.append(self._generate_quality_chart_script())
        
        return '\n'.join(scripts)
    
    def _generate_comparison_chart_script(self) -> str:
        """Generate comparison chart script."""
        if not self.comparison_report:
            return ""
        
        # Prepare data
        metrics = list(self.comparison_report.metric_comparisons.keys())
        models = self.comparison_report.model_names
        
        datasets = []
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
        
        for i, model in enumerate(models):
            data = []
            for metric_name in metrics:
                comparison = self.comparison_report.metric_comparisons[metric_name]
                if model in comparison.model_scores:
                    data.append(comparison.model_scores[model][0])
                else:
                    data.append(0)
            
            datasets.append({
                "label": model,
                "data": data,
                "backgroundColor": colors[i % len(colors)],
                "borderColor": colors[i % len(colors)],
                "borderWidth": 2
            })
        
        return f"""
        const comparisonCtx = document.getElementById('comparisonChart');
        if (comparisonCtx) {{
            new Chart(comparisonCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(metrics)},
                    datasets: {json.dumps(datasets)}
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Model Performance Comparison'
                        }},
                        legend: {{
                            display: true,
                            position: 'top'
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }}
                }}
            }});
        }}
        """
    
    def _generate_quality_chart_script(self) -> str:
        """Generate quality trends chart script."""
        if not self.quality_tracker:
            return ""
        
        # Prepare data
        datasets = []
        colors = ['#667eea', '#764ba2']
        
        for i, (metric_name, history) in enumerate(self.quality_tracker.metric_histories.items()):
            epochs = [e for e, _ in history.values]
            values = [v for _, v in history.values]
            
            datasets.append({
                "label": metric_name,
                "data": values,
                "borderColor": colors[i % len(colors)],
                "backgroundColor": colors[i % len(colors)] + '20',
                "borderWidth": 2,
                "fill": True,
                "tension": 0.4
            })
        
        # Get epochs from first metric
        if self.quality_tracker.metric_histories:
            first_history = list(self.quality_tracker.metric_histories.values())[0]
            epochs = [e for e, _ in first_history.values]
        else:
            epochs = []
        
        return f"""
        const qualityCtx = document.getElementById('qualityChart');
        if (qualityCtx) {{
            new Chart(qualityCtx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(epochs)},
                    datasets: {json.dumps(datasets)}
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Quality Trends Over Training'
                        }},
                        legend: {{
                            display: true,
                            position: 'top'
                        }}
                    }},
                    scales: {{
                        x: {{
                            title: {{
                                display: true,
                                text: 'Epoch'
                            }}
                        }},
                        y: {{
                            title: {{
                                display: true,
                                text: 'Metric Value'
                            }}
                        }}
                    }}
                }}
            }});
        }}
        """


# =========================
# Main (for testing)
# =========================

def main():
    """Test dashboard generator."""
    print("\n" + "=" * 80)
    print("DASHBOARD GENERATOR TEST")
    print("=" * 80)
    
    from src.evaluation.metrics import EvaluationResults
    from src.evaluation.comparison import ModelComparison, ComparisonReport
    from src.evaluation.quality_tracker import QualityTracker
    
    # Create dashboard
    dashboard = DashboardGenerator(
        title="Model Evaluation Dashboard",
        description="Comprehensive evaluation results for GPT models"
    )
    
    # Add evaluation results
    results_A = EvaluationResults()
    results_A.add_metric("accuracy", 0.85, "%", higher_is_better=True)
    results_A.add_metric("perplexity", 45.2, "", higher_is_better=False)
    results_A.add_metric("bleu-4", 68.5, "", higher_is_better=True)
    results_A.num_samples = 1000
    results_A.num_tokens = 50000
    
    results_B = EvaluationResults()
    results_B.add_metric("accuracy", 0.78, "%", higher_is_better=True)
    results_B.add_metric("perplexity", 52.1, "", higher_is_better=False)
    results_B.add_metric("bleu-4", 61.3, "", higher_is_better=True)
    results_B.num_samples = 1000
    results_B.num_tokens = 50000
    
    dashboard.add_evaluation_results("Model-A", results_A)
    dashboard.add_evaluation_results("Model-B", results_B)
    
    # Add comparison
    comparison = ModelComparison()
    comparison.add_model("Model-A", results_A)
    comparison.add_model("Model-B", results_B)
    report = comparison.compare()
    
    dashboard.add_comparison_report(report)
    
    # Add quality tracking
    tracker = QualityTracker(
        metrics_to_track=["accuracy", "perplexity"],
        regression_threshold=0.05
    )
    
    # Simulate training
    for epoch in range(1, 6):
        results = EvaluationResults()
        results.add_metric("accuracy", 0.70 + epoch * 0.03, "%", higher_is_better=True)
        results.add_metric("perplexity", 65.0 - epoch * 3.0, "", higher_is_better=False)
        tracker.add_evaluation(epoch, results)
    
    dashboard.add_quality_tracking(tracker)
    
    # Generate dashboard
    dashboard.generate("evaluation_dashboard_test.html")
    
    print("\n✓ Dashboard generated: evaluation_dashboard_test.html")
    print("  Open in browser to view")
    
    print("\n" + "=" * 80)
    print("✓ Dashboard generator test completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
