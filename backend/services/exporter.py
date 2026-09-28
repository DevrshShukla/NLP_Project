import io
import csv
import json
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import inch

from backend.models.schemas import AnalysisResult, ExportOptions
from backend.core.errors import create_app_error

class Exporter:
    @staticmethod
    def slugify(text: str) -> str:
        s = re.sub(r'[^\w\s-]', '', text).strip().lower()
        s = re.sub(r'[-\s]+', '-', s)
        return s or "document"

    @classmethod
    def export(
        cls,
        analysis: AnalysisResult,
        export_format: str,
        options: Optional[ExportOptions] = None
    ) -> Tuple[bytes, str, str]:
        """
        Returns (file_bytes, mime_type, filename)
        """
        fmt = export_format.lower().strip()
        opts = options or ExportOptions()
        
        # Filter topics if requested
        topics = analysis.topics
        if opts.priority_filter and opts.priority_filter != "ALL":
            topics = [t for t in topics if t.priority == opts.priority_filter]

        slug = cls.slugify(analysis.document.title)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        
        if fmt == "csv":
            return cls._export_csv(analysis, topics, slug, date_str, opts)
        elif fmt == "txt":
            return cls._export_txt(analysis, topics, slug, date_str, opts)
        elif fmt == "md":
            return cls._export_md(analysis, topics, slug, date_str, opts)
        elif fmt == "json":
            return cls._export_json(analysis, slug, date_str)
        elif fmt == "pdf":
            return cls._export_pdf(analysis, topics, slug, date_str, opts)
        else:
            raise create_app_error("INVALID_REQUEST", f"Unsupported export format: {export_format}")

    @classmethod
    def _export_csv(
        cls,
        analysis: AnalysisResult,
        topics: list,
        slug: str,
        date_str: str,
        opts: ExportOptions
    ) -> Tuple[bytes, str, str]:
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(["Rank", "Topic", "Score", "Priority", "Keywords", "Summary", "Occurrences"])
        
        for t in topics:
            # Formula-injection guard: prefix ' on cells starting with = + - @
            name_val = f"'{t.name}" if t.name.startswith(('=', '+', '-', '@')) else t.name
            summary_val = t.summary if opts.include_summaries else ""
            if summary_val.startswith(('=', '+', '-', '@')):
                summary_val = f"'{summary_val}"
            kw_val = "; ".join(t.keywords) if opts.include_keywords else ""
            if kw_val.startswith(('=', '+', '-', '@')):
                kw_val = f"'{kw_val}"
                
            writer.writerow([
                t.rank,
                name_val,
                f"{t.score:.1f}",
                t.priority,
                kw_val,
                summary_val,
                t.occurrences
            ])
            
        csv_bytes = output.getvalue().encode("utf-8-sig")  # UTF-8 with BOM for Excel
        filename = f"study-roadmap-{slug}-{date_str}.csv"
        return csv_bytes, "text/csv; charset=utf-8", filename

    @classmethod
    def _export_txt(
        cls,
        analysis: AnalysisResult,
        topics: list,
        slug: str,
        date_str: str,
        opts: ExportOptions
    ) -> Tuple[bytes, str, str]:
        lines = []
        lines.append("=" * 60)
        lines.append(f"STUDY TOPIC ROADMAP: {analysis.document.title.upper()}")
        lines.append("=" * 60)
        lines.append(f"Total Topics: {len(topics)} | HIGH: {analysis.stats.high} | MEDIUM: {analysis.stats.medium} | LOW: {analysis.stats.low}")
        lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("")
        
        for priority in ["HIGH", "MEDIUM", "LOW"]:
            group = [t for t in topics if t.priority == priority]
            if not group:
                continue
            lines.append("-" * 40)
            lines.append(f"[{priority} PRIORITY TOPICS]")
            lines.append("-" * 40)
            for t in group:
                lines.append(f"#{t.rank} {t.name} (Score: {t.score:.1f} | Occurrences: {t.occurrences})")
                if opts.include_summaries and t.summary:
                    lines.append(f"   Summary: {t.summary}")
                if opts.include_keywords and t.keywords:
                    lines.append(f"   Keywords: {', '.join(t.keywords)}")
                lines.append("")
                
        if opts.include_keywords and analysis.keywords:
            lines.append("-" * 40)
            lines.append("[DOCUMENT KEYWORDS]")
            lines.append("-" * 40)
            kw_strs = [f"{kw.term} ({kw.count})" for kw in analysis.keywords[:25]]
            lines.append(", ".join(kw_strs))
            lines.append("")

        txt_bytes = "\n".join(lines).encode("utf-8")
        filename = f"study-roadmap-{slug}-{date_str}.txt"
        return txt_bytes, "text/plain; charset=utf-8", filename

    @classmethod
    def _export_md(
        cls,
        analysis: AnalysisResult,
        topics: list,
        slug: str,
        date_str: str,
        opts: ExportOptions
    ) -> Tuple[bytes, str, str]:
        lines = []
        lines.append(f"# Study Topic Roadmap: {analysis.document.title}\n")
        lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ")
        lines.append(f"**Topics Found:** {len(topics)} (HIGH: {analysis.stats.high}, MEDIUM: {analysis.stats.medium}, LOW: {analysis.stats.low})\n")
        
        for priority in ["HIGH", "MEDIUM", "LOW"]:
            group = [t for t in topics if t.priority == priority]
            if not group:
                continue
            lines.append(f"## {priority} Priority\n")
            for t in group:
                lines.append(f"### {t.rank}. {t.name} *(Score: {t.score:.1f})*")
                if opts.include_summaries and t.summary:
                    lines.append(f"> {t.summary}\n")
                if opts.include_keywords and t.keywords:
                    lines.append(f"- **Key concepts:** {', '.join(t.keywords)}")
                lines.append("")
                
        if opts.include_keywords and analysis.keywords:
            lines.append("## Important Keywords\n")
            chips = [f"`{kw.term}`" for kw in analysis.keywords[:30]]
            lines.append(" ".join(chips) + "\n")

        md_bytes = "\n".join(lines).encode("utf-8")
        filename = f"study-roadmap-{slug}-{date_str}.md"
        return md_bytes, "text/markdown; charset=utf-8", filename

    @classmethod
    def _export_json(
        cls,
        analysis: AnalysisResult,
        slug: str,
        date_str: str
    ) -> Tuple[bytes, str, str]:
        json_str = analysis.model_dump_json(indent=2)
        filename = f"study-roadmap-{slug}-{date_str}.json"
        return json_str.encode("utf-8"), "application/json", filename

    @classmethod
    def _export_pdf(
        cls,
        analysis: AnalysisResult,
        topics: list,
        slug: str,
        date_str: str,
        opts: ExportOptions
    ) -> Tuple[bytes, str, str]:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=colors.HexColor('#1E1B4B')
        )
        subtitle_style = ParagraphStyle(
            'DocSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#6B7280')
        )
        h2_style = ParagraphStyle(
            'SectionH2',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=13,
            leading=17,
            textColor=colors.HexColor('#312E81'),
            spaceBefore=12,
            spaceAfter=6
        )
        cell_style = ParagraphStyle(
            'CellText',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=12,
            textColor=colors.HexColor('#1F2937')
        )
        cell_bold = ParagraphStyle(
            'CellBold',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=9,
            leading=12,
            textColor=colors.HexColor('#111827')
        )

        elements = []

        # Title & Subtitle
        elements.append(Paragraph(f"Study Topic Roadmap: {analysis.document.title}", title_style))
        meta_text = (
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | "
            f"Words: {analysis.document.word_count} | Sections: {analysis.document.section_count} | "
            f"Topics: {len(topics)} (HIGH: {analysis.stats.high}, MED: {analysis.stats.medium}, LOW: {analysis.stats.low})"
        )
        elements.append(Paragraph(meta_text, subtitle_style))
        elements.append(Spacer(1, 10))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E5E7EB'), spaceBefore=5, spaceAfter=15))

        # Topics Table
        table_data = [[
            Paragraph("Rank", cell_bold),
            Paragraph("Topic & Summary", cell_bold),
            Paragraph("Score", cell_bold),
            Paragraph("Priority", cell_bold)
        ]]

        for t in topics:
            content_parts = [f"<b>{t.name}</b>"]
            if opts.include_summaries and t.summary:
                content_parts.append(f"<font color='#4B5563' size='8'>{t.summary}</font>")
            if opts.include_keywords and t.keywords:
                content_parts.append(f"<font color='#6366F1' size='7.5'>Keywords: {', '.join(t.keywords)}</font>")
                
            p_color = '#DC2626' if t.priority == 'HIGH' else ('#D97706' if t.priority == 'MEDIUM' else '#0284C7')
            priority_cell = f"<font color='{p_color}'><b>{t.priority}</b></font>"

            table_data.append([
                Paragraph(f"#{t.rank}", cell_style),
                Paragraph("<br/>".join(content_parts), cell_style),
                Paragraph(f"{t.score:.1f}", cell_style),
                Paragraph(priority_cell, cell_style)
            ])

        col_widths = [0.6 * inch, 5.0 * inch, 0.8 * inch, 0.9 * inch]
        t_table = Table(table_data, colWidths=col_widths, repeatRows=1)
        t_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F3F4F6')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
        ]))
        elements.append(t_table)

        # Document Keywords section
        if opts.include_keywords and analysis.keywords:
            elements.append(Spacer(1, 15))
            elements.append(Paragraph("Document Keywords", h2_style))
            kw_terms = ", ".join([f"{kw.term} ({kw.count})" for kw in analysis.keywords[:25]])
            elements.append(Paragraph(kw_terms, cell_style))

        doc.build(elements)
        pdf_bytes = buffer.getvalue()
        filename = f"study-roadmap-{slug}-{date_str}.pdf"
        return pdf_bytes, "application/pdf", filename
