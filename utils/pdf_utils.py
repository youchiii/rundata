import io
import textwrap

import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas

canvas = None
FONT_REGISTERED = False
FONT_NAME = 'IPAexGothic'
FONT_PATH = 'ipaexm.ttf'

try:
    canvas = rl_canvas
    pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
    FONT_REGISTERED = True
except Exception as e:
    st.warning(f"日本語フォントの登録に失敗しました: {e}。PDFで文字化けする可能性があります。")
    FONT_REGISTERED = False

def make_pdf(content_list: list) -> io.BytesIO | None:
    """与えられたコンテンツリスト（文字列またはPlotly Figure）をPDFにして返す。"""
    if canvas is None:
        return None
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    line_height = 14

    if FONT_REGISTERED:
        c.setFont(FONT_NAME, 12)
        line_height = 14
    else:
        c.setFont('Helvetica', 12)

    y = height - 40

    for item in content_list:
        if isinstance(item, str):
            wrap_width = int((width - 80) / (6 if FONT_REGISTERED else 8))
            wrapped_lines = textwrap.wrap(item, width=wrap_width)
            for w_line in wrapped_lines:
                if y < 40:
                    c.showPage()
                    y = height - 40
                    if FONT_REGISTERED:
                        c.setFont(FONT_NAME, 12)
                c.drawString(40, y, w_line)
                y -= line_height
            y -= line_height
        elif hasattr(item, 'write_image'):
            img_buffer = io.BytesIO()
            try:
                original_width = item.layout.width if item.layout.width else 700
                original_height = item.layout.height if item.layout.height else 450
                img_width = 500
                img_height = int(original_height * (img_width / original_width)) if original_width else 300
                
                item.write_image(img_buffer, format='png', width=img_width, height=img_height, scale=1)
                img_buffer.seek(0)
                img = ImageReader(img_buffer)

                if y - img_height < 40:
                    c.showPage()
                    y = height - 40
                    if FONT_REGISTERED:
                        c.setFont(FONT_NAME, 12)
                
                c.drawImage(img, (width - img_width) / 2, y - img_height, width=img_width, height=img_height)
                y -= (img_height + line_height)
            except Exception as e:
                st.warning(f"グラフのPDF埋め込みに失敗しました: {e}。`kaleido`がインストールされているか確認してください。")
                if y < 40:
                    c.showPage()
                    y = height - 40
                    if FONT_REGISTERED:
                        c.setFont(FONT_NAME, 12)
                c.drawString(40, y, f"[グラフの埋め込みに失敗しました: {item.layout.title.text if item.layout.title else '無題のグラフ'}]")
                y -= line_height
        else:
            if y < 40:
                c.showPage()
                y = height - 40
                if FONT_REGISTERED:
                    c.setFont(FONT_NAME, 12)
            c.drawString(40, y, str(item))
            y -= line_height

    c.save()
    buffer.seek(0)
    return buffer
