import streamlit as st
import pandas as pd
import numpy as np
import spacy
import ginza
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
import io
import base64
import jaconv
import re
from pyvis.network import Network
import tempfile
import os

# ページ設定
st.set_page_config(
    page_title="テキスト分析アプリ",
    page_icon="📊",
    layout="wide"
)

# フォントパス（Streamlit Cloud対応）
import os
FONT_PATH = os.path.join(os.path.dirname(__file__), "font", "NotoSansJP-VariableFont_wght.ttf")

@st.cache_resource
def load_spacy_model():
    """spaCyモデルを読み込み"""
    try:
        nlp = spacy.load("ja_ginza")
        return nlp
    except OSError:
        st.error("GINZAモデルが見つかりません。インストールしてください：pip install ja-ginza")
        return None

@st.cache_data
def load_uploaded_data(uploaded_file):
    """アップロードされたCSVデータを読み込み"""
    try:
        if uploaded_file is not None:
            # ファイルの拡張子を確認
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                return df
            else:
                st.error("CSVファイルをアップロードしてください。")
                return None
        else:
            return None
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return None

@st.cache_data
def load_default_data():
    """デフォルトCSVデータを読み込み（フォールバック用）"""
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "csv", "data.csv")
        if os.path.exists(csv_path):
            # 複数のエンコーディングを試行
            encodings = ['utf-8', 'shift_jis', 'cp932', 'utf-8-sig']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding)
                    st.write(f"✅ CSVファイルを読み込みました（エンコーディング: {encoding}）")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                st.error("CSVファイルのエンコーディングを判別できませんでした。")
                return None
                
            return df
        else:
            st.warning(f"デフォルトデータファイルが見つかりません: {csv_path}")
            return None
    except Exception as e:
        st.error(f"デフォルトデータ読み込みエラー: {e}")
        return None

def preprocess_text(text):
    """テキストの前処理"""
    if pd.isna(text):
        return ""
    
    # 改行とタブを削除
    text = re.sub(r'[\n\t\r]', ' ', str(text))
    # 連続する空白を1つに
    text = re.sub(r'\s+', ' ', text)
    # 全角・半角を統一
    text = jaconv.normalize(text, 'NFKC')
    return text.strip()

def extract_keywords(nlp, texts, min_freq=2, pos_tags=['NOUN', 'PROPN', 'ADJ']):
    """形態素解析でキーワードを抽出"""
    all_words = []
    word_pairs = []
    processed_texts = 0
    total_tokens = 0
    filtered_tokens = 0
    
    for text in texts:
        if pd.isna(text) or text.strip() == "":
            continue
            
        processed_texts += 1
        doc = nlp(text)
        words = []
        
        for token in doc:
            total_tokens += 1
            # 品詞フィルタリング
            if token.pos_ in pos_tags and not token.is_stop and len(token.text) > 1:
                # 基本形を使用
                lemma = token.lemma_.lower()
                # 数字や記号のみは除外（日本語文字を含む場合は許可）
                if not re.match(r'^[a-zA-Z0-9\W]+$', lemma) and len(lemma) >= 2:
                    words.append(lemma)
                    all_words.append(lemma)
                    filtered_tokens += 1
        
        # 共起ペアを抽出（窓幅2）
        for i in range(len(words) - 1):
            for j in range(i + 1, min(i + 3, len(words))):
                if words[i] != words[j]:
                    word_pairs.append((words[i], words[j]))
    
    # デバッグ情報
    st.write(f"🔧 処理統計:")
    st.write(f"  - 処理したテキスト数: {processed_texts}")
    st.write(f"  - 総トークン数: {total_tokens}")
    st.write(f"  - フィルタリング後トークン数: {filtered_tokens}")
    st.write(f"  - 抽出前の単語数: {len(all_words)}")
    
    # 頻度フィルタリング
    word_freq = Counter(all_words)
    filtered_words = {word: freq for word, freq in word_freq.items() if freq >= min_freq}
    
    # 共起ペアの頻度計算
    pair_freq = Counter(word_pairs)
    filtered_pairs = {pair: freq for pair, freq in pair_freq.items() if freq >= min_freq}
    
    st.write(f"  - 最小出現回数({min_freq})フィルタリング後: {len(filtered_words)}")
    
    return filtered_words, filtered_pairs

def create_wordcloud(word_freq, font_path):
    """ワードクラウドを生成"""
    if not word_freq:
        return None
    
    try:
        # フォントファイルの存在確認
        if not os.path.exists(font_path):
            st.warning(f"フォントファイルが見つかりません: {font_path}")
            # デフォルトフォントを使用
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=100,
                relative_scaling=0.5,
                random_state=42,
                prefer_horizontal=0.9,
                collocations=False,
                regexp=r"[\w\-\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]+"
            ).generate_from_frequencies(word_freq)
        else:
            # 日本語フォントを使用
            wordcloud = WordCloud(
                font_path=font_path,
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=100,
                relative_scaling=0.5,
                random_state=42,
                prefer_horizontal=0.9,
                collocations=False,
                regexp=r"[\w\-\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]+"
            ).generate_from_frequencies(word_freq)
        
        return wordcloud
    except Exception as e:
        st.error(f"ワードクラウド生成エラー: {e}")
        return None

def create_cooccurrence_network(word_pairs, word_freq, min_freq=2):
    """共起ネットワークを作成"""
    if not word_pairs:
        return None
    
    G = nx.Graph()
    
    # ノードを追加
    for word, freq in word_freq.items():
        if freq >= min_freq:
            G.add_node(word, size=freq, freq=freq)
    
    # エッジを追加
    for (word1, word2), freq in word_pairs.items():
        if word1 in G and word2 in G:
            G.add_edge(word1, word2, weight=freq)
    
    return G

def plot_network_plotly(G, title="共起ネットワーク"):
    """Plotlyでネットワークを可視化"""
    if not G or len(G.nodes()) == 0:
        return None
    
    # レイアウト計算
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # エッジの情報
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        weight = G[edge[0]][edge[1]]['weight']
        edge_info.append(f"{edge[0]} - {edge[1]}<br>共起回数: {weight}")
    
    # ノードの情報
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        freq = G.nodes[node]['freq']
        node_text.append(f"{node}<br>出現回数: {freq}")
        node_size.append(max(10, freq * 2))
        node_color.append(freq)
    
    # エッジのプロット
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # ノードのプロット
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node for node in G.nodes()],
        textposition="middle center",
        hovertext=node_text,
        # インタラクティブな設定
        selected=dict(
            marker=dict(
                color='red',
                size=15
            )
        ),
        unselected=dict(
            marker=dict(
                opacity=0.8
            )
        ),
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            reversescale=True,
            color=node_color,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title="出現回数",
                xanchor="left"
            ),
            line=dict(width=2, color='black'),
            opacity=1.0
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=dict(text=title, font=dict(size=16)),
                       showlegend=False,
                       hovermode='closest',
                       dragmode='pan',  # ドラッグモードを設定
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="ノードサイズは出現回数、色は共起の強さを表します。マウスドラッグで移動、ホイールでズーム可能",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor="left", yanchor="bottom",
                           font=dict(color="gray", size=12)
                       )],
                       xaxis=dict(
                           showgrid=False, 
                           zeroline=False, 
                           showticklabels=False,
                           scaleanchor="y",  # x軸とy軸のスケールを統一
                           scaleratio=1,
                           constrain="domain"
                       ),
                       yaxis=dict(
                           showgrid=False, 
                           zeroline=False, 
                           showticklabels=False,
                           constrain="domain"
                       ),
                       plot_bgcolor='white',
                       # インタラクティブな設定
                       clickmode='event+select',
                       selectdirection='d'
                   ))
    
    return fig

def plot_network_pyvis(G, title="共起ネットワーク", height="600px", width="100%"):
    """pyvisでネットワークを可視化"""
    if not G or len(G.nodes()) == 0:
        return None
    
    try:
        # pyvisネットワークを作成
        net = Network(height=height, width=width, bgcolor="#222222", font_color="white")
        
        # ノードとエッジを追加
        for node in G.nodes():
            freq = G.nodes[node]['freq']
            # ノードサイズを頻度に基づいて調整
            size = max(10, freq * 3)
            # ノードの色を頻度に基づいて設定
            color_intensity = min(255, freq * 50)
            color = f"rgb({color_intensity}, {255-color_intensity}, 100)"
            
            net.add_node(
                node, 
                label=node,
                size=size,
                color=color,
                title=f"単語: {node}<br>出現回数: {freq}"
            )
        
        # エッジを追加
        for edge in G.edges():
            weight = G[edge[0]][edge[1]]['weight']
            # エッジの太さを重みに基づいて調整
            width = max(1, weight * 2)
            net.add_edge(edge[0], edge[1], width=width, title=f"共起回数: {weight}")
        
        # 物理エンジンの設定
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09,
              "avoidOverlap": 0.1
            }
          },
          "interaction": {
            "hover": true,
            "hoverConnectedEdges": true,
            "selectConnectedEdges": false
          },
          "nodes": {
            "font": {
              "size": 14,
              "color": "white"
            },
            "borderWidth": 2,
            "borderWidthSelected": 4
          },
          "edges": {
            "smooth": {
              "enabled": true,
              "type": "continuous"
            },
            "color": {
              "color": "rgba(255,255,255,0.5)",
              "highlight": "rgba(255,255,255,1)"
            }
          }
        }
        """)
        
        return net
        
    except Exception as e:
        st.error(f"pyvisネットワーク生成エラー: {e}")
        return None

def plot_network_matplotlib(G, layout_type='spring', title="共起ネットワーク", figsize=(12, 8)):
    """Matplotlibでネットワークを可視化"""
    if not G or len(G.nodes()) == 0:
        return None
    
    # 日本語フォント設定
    try:
        font_path = FONT_PATH
        # フォントプロパティを作成
        font_prop = fm.FontProperties(fname=font_path)
        # matplotlibのデフォルトフォントに設定
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止
    except Exception as e:
        font_prop = None
        plt.rcParams['font.family'] = 'DejaVu Sans'
        st.warning(f"日本語フォントの設定に失敗しました: {e}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # レイアウト計算
    layout_functions = {
        'spring': nx.spring_layout,
        'circular': nx.circular_layout,
        'random': nx.random_layout,
        'shell': nx.shell_layout,
        'kamada_kawai': nx.kamada_kawai_layout,
        'spectral': nx.spectral_layout,
        'bipartite': nx.bipartite_layout,
        'planar': nx.planar_layout
    }
    
    if layout_type in layout_functions:
        try:
            if layout_type == 'bipartite':
                # 二部グラフレイアウトの場合は特別処理
                pos = nx.bipartite_layout(G, list(G.nodes())[:len(G.nodes())//2])
            elif layout_type == 'planar':
                # 平面グラフレイアウトの場合は特別処理
                pos = nx.planar_layout(G) if nx.is_planar(G) else nx.spring_layout(G)
            else:
                pos = layout_functions[layout_type](G)
        except Exception as e:
            st.warning(f"{layout_type}レイアウトでエラーが発生しました。springレイアウトを使用します。")
            pos = nx.spring_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # ノードサイズと色を計算
    node_sizes = []
    node_colors = []
    node_labels = {}
    
    for node in G.nodes():
        freq = G.nodes[node]['freq']
        # ノードサイズを頻度に基づいて調整
        size = max(100, freq * 50)
        node_sizes.append(size)
        node_colors.append(freq)
        node_labels[node] = node
    
    # エッジを描画
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray', width=0.5, ax=ax)
    
    # ノードを描画
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        alpha=0.8,
        ax=ax
    )
    
    # ノードラベルを描画
    nx.draw_networkx_labels(
        G, pos, 
        labels=node_labels,
        font_size=8,
        font_weight='bold',
        font_properties=font_prop if font_prop else None,
        ax=ax
    )
    
    # カラーバーを追加
    if nodes:
        plt.colorbar(nodes, ax=ax, label='出現回数')
    
    # グラフの設定
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # レイアウト情報を表示
    layout_info = {
        'spring': 'Spring Layout (バネモデル)',
        'circular': 'Circular Layout (円形)',
        'random': 'Random Layout (ランダム)',
        'shell': 'Shell Layout (同心円)',
        'kamada_kawai': 'Kamada-Kawai Layout',
        'spectral': 'Spectral Layout',
        'bipartite': 'Bipartite Layout (二部グラフ)',
        'planar': 'Planar Layout (平面グラフ)'
    }
    
    ax.text(0.02, 0.98, f'レイアウト: {layout_info.get(layout_type, layout_type)}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ネットワーク統計を表示
    stats_text = f'ノード数: {len(G.nodes())}\nエッジ数: {len(G.edges())}\n密度: {nx.density(G):.3f}'
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig

def setup_japanese_font():
    """日本語フォントの設定"""
    try:
        font_path = FONT_PATH
        if os.path.exists(font_path):
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return True
        else:
            st.warning(f"フォントファイルが見つかりません: {font_path}")
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
            return False
    except Exception as e:
        st.warning(f"日本語フォントの設定に失敗しました: {e}")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        return False

def main():
    st.title("📊 テキスト分析アプリ")
    st.markdown("---")
    
    # 日本語フォント設定
    font_setup_success = setup_japanese_font()
    
    # データアップロード
    st.subheader("📁 データアップロード")
    
    # ファイルアップローダー
    uploaded_file = st.file_uploader(
        "CSVファイルをアップロードしてください",
        type=['csv'],
        help="テキスト分析用のCSVファイルをアップロードしてください。テキスト列を含む必要があります。"
    )
    
    # データ読み込み（デフォルトでdata.csvを読み込み）
    df = load_default_data()
    
    if df is not None:
        st.success("📋 デフォルトデータ（data.csv）を読み込みました。")
        st.info("💡 独自のデータを使用する場合は、上記からCSVファイルをアップロードしてください。")
    else:
        st.warning("⚠️ デフォルトデータが見つかりません。CSVファイルをアップロードしてください。")
    
    # アップロードされたファイルがある場合は上書き
    if uploaded_file is not None:
        uploaded_df = load_uploaded_data(uploaded_file)
        if uploaded_df is not None:
            df = uploaded_df
            st.success("📁 アップロードされたCSVファイルを読み込みました。")
    
    if df is None:
        st.error("データを読み込めませんでした。CSVファイルをアップロードしてください。")
        return
    
    # spaCyモデル読み込み
    nlp = load_spacy_model()
    if nlp is None:
        return
    
    st.sidebar.header("⚙️ 設定")
    
    # データ表示
    st.subheader("📋 データ概要")
    
    # 現在のデータソースを表示
    if uploaded_file is not None:
        st.info(f"📁 現在のデータソース: アップロードファイル「{uploaded_file.name}」")
    else:
        st.info("📋 現在のデータソース: デフォルトデータ（data.csv）")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("総行数", len(df))
    
    with col2:
        st.metric("総列数", len(df.columns))
        st.caption(f"列名: {', '.join(df.columns.tolist())}")
    
    with col3:
        # テキスト列の選択（より柔軟な検出）
        text_columns = []
        for col in df.columns:
            # 文字列型または数値型でも文字列として扱える列を検出
            if (df[col].dtype == 'object' or 
                (df[col].dtype in ['int64', 'float64'] and df[col].astype(str).str.len().mean() > 5)):
                text_columns.append(col)
        
        # 明示的に'text'列がある場合は優先
        if 'text' in df.columns and 'text' not in text_columns:
            text_columns.insert(0, 'text')
        
        # デバッグ情報を表示
        st.write(f"🔍 検出された列: {list(df.columns)}")
        st.write(f"🔍 データ型: {dict(df.dtypes)}")
        st.write(f"🔍 テキスト列候補: {text_columns}")
        
        if text_columns:
            # デフォルトで'text'列を選択（存在する場合）
            default_index = 0
            if 'text' in text_columns:
                default_index = text_columns.index('text')
            
            selected_text_column = st.selectbox(
                "分析対象のテキスト列を選択",
                text_columns,
                index=default_index,
                help="テキスト分析を行う列を選択してください"
            )
            
            # 選択された列の有効テキスト数を計算
            non_empty_texts = df[selected_text_column].dropna().astype(str).str.strip()
            non_empty_texts = non_empty_texts[non_empty_texts != '']
            st.metric("有効テキスト数", len(non_empty_texts))
            
            # 選択された列のサンプルを表示
            st.write(f"📝 選択された列「{selected_text_column}」のサンプル:")
            sample_data = df[selected_text_column].dropna().head(3).tolist()
            for i, text in enumerate(sample_data, 1):
                st.write(f"  {i}. {str(text)[:50]}...")
        else:
            st.warning("テキスト列が見つかりません")
            st.write("利用可能な列とそのデータ型:")
            for col in df.columns:
                st.write(f"  - {col}: {df[col].dtype}")
            return
    
    # データプレビュー
    st.subheader("👀 データプレビュー")
    if st.checkbox("データを表示", value=False):
        st.dataframe(df.head(10), use_container_width=True)
        
        # 選択されたテキスト列のサンプル表示
        if 'selected_text_column' in locals():
            st.subheader(f"📝 {selected_text_column}列のサンプル")
            sample_texts = df[selected_text_column].dropna().head(5).tolist()
            for i, text in enumerate(sample_texts, 1):
                st.write(f"{i}. {text[:100]}{'...' if len(str(text)) > 100 else ''}")
    
    # 設定パラメータ
    min_freq = st.sidebar.slider("最小出現回数", 1, 10, 1)
    max_words = st.sidebar.slider("最大単語数", 50, 200, 100)
    
    # 品詞選択
    pos_options = {
        '名詞': 'NOUN',
        '固有名詞': 'PROPN', 
        '形容詞': 'ADJ',
        '動詞': 'VERB'
    }
    
    selected_pos = st.sidebar.multiselect(
        "分析対象の品詞",
        options=list(pos_options.keys()),
        default=['名詞', '固有名詞', '形容詞', '動詞']
    )
    
    selected_pos_tags = [pos_options[pos] for pos in selected_pos]
    
    # テキスト処理
    if 'selected_text_column' in locals():
        texts = df[selected_text_column].apply(preprocess_text).tolist()
        
        with st.spinner("形態素解析を実行中..."):
            # デバッグ情報を表示
            st.write(f"📊 処理対象テキスト数: {len(texts)}")
            st.write(f"📝 最初の3つのテキストサンプル:")
            for i, text in enumerate(texts[:3], 1):
                st.write(f"  {i}. {text[:100]}{'...' if len(text) > 100 else ''}")
            
            word_freq, word_pairs = extract_keywords(
                nlp, texts, min_freq, selected_pos_tags
            )
            
            # 抽出結果のデバッグ情報
            st.write(f"🔍 抽出された単語数: {len(word_freq)}")
            st.write(f"🔗 抽出された共起ペア数: {len(word_pairs)}")
            
            if word_freq:
                st.write("📈 上位10単語:")
                top_words = dict(Counter(word_freq).most_common(10))
                for word, freq in top_words.items():
                    st.write(f"  - {word}: {freq}回")
            else:
                st.warning("⚠️ 単語が抽出されませんでした。以下を確認してください：")
                st.write("- テキストに日本語が含まれているか")
                st.write("- 最小出現回数の設定が高すぎないか")
                st.write("- 選択した品詞が適切か")
        
        # 結果表示
        st.subheader("📈 分析結果")
        
        if word_freq:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔤 ワードクラウド")
                
                # ワードクラウド生成
                wordcloud = create_wordcloud(word_freq, FONT_PATH)
                if wordcloud:
                    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                    ax_wc.imshow(wordcloud, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)
                    
                    # ダウンロードボタン
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
                    img_buffer.seek(0)
                    
                    st.download_button(
                        label="ワードクラウドをダウンロード",
                        data=img_buffer.getvalue(),
                        file_name="wordcloud.png",
                        mime="image/png"
                    )
                else:
                    st.warning("ワードクラウドを生成できませんでした。")
            
            with col2:
                st.subheader("📊 頻出単語トップ20")
                
                # 頻出単語表示
                top_words = dict(Counter(word_freq).most_common(20))
                
                for i, (word, freq) in enumerate(top_words.items(), 1):
                    st.write(f"{i:2d}. **{word}** ({freq}回)")
            
            # 共起ネットワーク
            st.subheader("🕸️ 共起ネットワーク")
            
            if word_pairs:
                G = create_cooccurrence_network(word_pairs, word_freq, min_freq)
                if G and len(G.nodes()) > 0:
                    # 可視化方法を選択
                    viz_type = st.radio(
                        "可視化方法を選択",
                        ["pyvis (高度なインタラクティブ)", "Plotly (インタラクティブ)", "Matplotlib (静的)"],
                        horizontal=True
                    )
                    
                    if viz_type == "pyvis (高度なインタラクティブ)":
                        # pyvisネットワークの設定
                        col1, col2 = st.columns(2)
                        with col1:
                            height = st.selectbox("表示高さ", ["400px", "500px", "600px", "700px", "800px"], index=2)
                        with col2:
                            physics = st.selectbox("物理エンジン", ["有効", "無効"], index=0)
                        
                        # pyvisネットワークを生成
                        with st.spinner("pyvisネットワークを生成中..."):
                            net = plot_network_pyvis(G, height=height)
                        
                        if net:
                            # HTMLファイルとして保存
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                                net.save_graph(f.name)
                                html_file = f.name
                            
                            try:
                                # HTMLファイルを読み込んで表示
                                with open(html_file, 'r', encoding='utf-8') as f:
                                    html_content = f.read()
                                
                                # 物理エンジンの設定を動的に変更
                                if physics == "無効":
                                    html_content = html_content.replace('"enabled": true', '"enabled": false')
                                
                                st.components.v1.html(html_content, height=int(height.replace('px', '')) + 50)
                                
                            finally:
                                # 一時ファイルを削除
                                try:
                                    os.unlink(html_file)
                                except OSError:
                                    pass  # ファイルが既に削除されている場合は無視
                        else:
                            st.warning("pyvisネットワークの可視化に失敗しました。")
                    
                    elif viz_type == "Plotly (インタラクティブ)":
                        fig_net = plot_network_plotly(G, "単語の共起ネットワーク")
                        if fig_net:
                            st.plotly_chart(fig_net, use_container_width=True)
                        else:
                            st.warning("Plotlyネットワークの可視化に失敗しました。")
                    
                    else:  # Matplotlib
                        # レイアウト選択
                        layout_options = {
                            'Spring Layout (バネモデル)': 'spring',
                            'Circular Layout (円形)': 'circular',
                            'Random Layout (ランダム)': 'random',
                            'Shell Layout (同心円)': 'shell',
                            'Kamada-Kawai Layout': 'kamada_kawai',
                            'Spectral Layout': 'spectral',
                            'Bipartite Layout (二部グラフ)': 'bipartite',
                            'Planar Layout (平面グラフ)': 'planar'
                        }
                        
                        selected_layout = st.selectbox(
                            "レイアウトを選択",
                            options=list(layout_options.keys()),
                            index=0
                        )
                        
                        layout_type = layout_options[selected_layout]
                        
                        # 図のサイズ設定
                        fig_size = st.slider("図のサイズ", 8, 16, 12)
                        
                        # Matplotlibネットワークを描画
                        with st.spinner(f"{selected_layout}でネットワークを生成中..."):
                            fig_matplotlib = plot_network_matplotlib(
                                G, 
                                layout_type=layout_type,
                                title=f"単語の共起ネットワーク - {selected_layout}",
                                figsize=(fig_size, fig_size * 0.7)
                            )
                        
                        if fig_matplotlib:
                            st.pyplot(fig_matplotlib)
                            
                            # ダウンロードボタン
                            img_buffer = io.BytesIO()
                            fig_matplotlib.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                            img_buffer.seek(0)
                            
                            st.download_button(
                                label="ネットワーク図をダウンロード",
                                data=img_buffer.getvalue(),
                                file_name=f"network_{layout_type}.png",
                                mime="image/png"
                            )
                        else:
                            st.warning("Matplotlibネットワークの可視化に失敗しました。")
                    
                    # ネットワーク統計（両方の可視化で共通）
                    st.subheader("📊 ネットワーク統計")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("ノード数", len(G.nodes()))
                    with col2:
                        st.metric("エッジ数", len(G.edges()))
                    with col3:
                        st.metric("密度", f"{nx.density(G):.3f}")
                    with col4:
                        if len(G.nodes()) > 0:
                            degree_centrality = nx.degree_centrality(G)
                            max_centrality_node = max(degree_centrality, key=degree_centrality.get)
                            st.metric("中心性最大", max_centrality_node)
                    
                    # 中心性分析
                    if len(G.nodes()) > 0:
                        st.subheader("🎯 中心性分析")
                        
                        # 各種中心性を計算
                        degree_centrality = nx.degree_centrality(G)
                        betweenness_centrality = nx.betweenness_centrality(G)
                        closeness_centrality = nx.closeness_centrality(G)
                        
                        # 上位5位を表示
                        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                        top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                        top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**次数中心性**")
                            for i, (node, centrality) in enumerate(top_degree, 1):
                                st.write(f"{i}. {node}: {centrality:.3f}")
                        
                        with col2:
                            st.write("**媒介中心性**")
                            for i, (node, centrality) in enumerate(top_betweenness, 1):
                                st.write(f"{i}. {node}: {centrality:.3f}")
                        
                        with col3:
                            st.write("**近接中心性**")
                            for i, (node, centrality) in enumerate(top_closeness, 1):
                                st.write(f"{i}. {node}: {centrality:.3f}")
                        
                else:
                    st.warning("共起ネットワークを生成できませんでした。")
            else:
                st.warning("共起ペアが見つかりませんでした。")
        else:
            st.warning("分析対象の単語が見つかりませんでした。設定を調整してください。")
    else:
        st.error("テキスト列が選択されていません。")
    
    # フッター
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        📊 テキスト分析アプリ | Powered by Streamlit, GINZA, WordCloud, NetworkX
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
