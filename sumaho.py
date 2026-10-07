import csv
import io
import json
import re
import urllib.parse
import urllib.request
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# ==========================================
# 1. Web API 住所・施設検索部
# ==========================================


def search_address_to_latlon(address_text):
  """住所・施設名から緯度経度(lat, lon)を取得"""
  if not address_text or not address_text.strip():
    return None, None

  query = address_text.strip()
  encoded_query = urllib.parse.quote(query)

  # 1. 国土地理院 API
  try:
    gsi_url = f'https://msearch.gsi.go.jp/address-search/AddressSearch?q={encoded_query}'
    req = urllib.request.Request(
        gsi_url, headers={'User-Agent': 'TSP-Field-App'}
    )
    with urllib.request.urlopen(req, timeout=5) as response:
      data = json.loads(response.read().decode())
      if data and len(data) > 0 and 'geometry' in data[0]:
        coords = data[0]['geometry']['coordinates']
        return coords[1], coords[0]  # lat, lon
  except Exception:
    pass

  # 2. OpenStreetMap (Nominatim) API
  try:
    osm_url = f'https://nominatim.openstreetmap.org/search?q={encoded_query}&format=json&limit=1'
    req = urllib.request.Request(
        osm_url, headers={'User-Agent': 'TSP-Field-App'}
    )
    with urllib.request.urlopen(req, timeout=5) as response:
      data = json.loads(response.read().decode())
      if data and len(data) > 0:
        return float(data[0]['lat']), float(data[0]['lon'])
  except Exception:
    pass

  return None, None


def parse_lat_lon_pair(val1, val2=None):
  """単一文字列または2値から緯度経度を安全に取得"""
  if val1 is None or str(val1).strip() == '' or str(val1).strip().lower() == 'nan':
    return None, None
  try:
    if (
        val2 is None
        or str(val2).strip() == ''
        or str(val2).strip().lower() == 'nan'
    ):
      nums = re.findall(r'[-+]?\d*\.\d+|\d+', str(val1))
      if len(nums) < 2:
        return None, None
      n1, n2 = float(nums[0]), float(nums[1])
    else:
      n1, n2 = float(val1), float(val2)

    if 120 <= n1 <= 155 and 20 <= n2 <= 50:
      return n2, n1
    elif 20 <= n1 <= 50 and 120 <= n2 <= 155:
      return n1, n2
    else:
      return n1, n2
  except Exception:
    return None, None


def extract_gdrive_file_id(url):
  match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
  if match:
    return match.group(1)
  match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
  if match:
    return match.group(1)
  return None


def download_from_gdrive(file_id):
  download_url = (
      f'https://drive.google.com/uc?export=download&id={file_id}&confirm=t'
  )
  req = urllib.request.Request(
      download_url, headers={'User-Agent': 'Mozilla/5.0'}
  )
  with urllib.request.urlopen(req, timeout=15) as response:
    content = response.read()
  return content


def parse_dataframe_to_spots(df):
  header = [str(c).strip() for c in df.columns]
  name_idx, latlon_idx, lat_idx, lon_idx = -1, -1, -1, -1

  for idx, col in enumerate(header):
    col_lower = col.lower()
    if any(
        k in col_lower
        for k in ['管理番号', '名称', '名前', 'name', 'id', 'スポット']
    ):
      if name_idx == -1:
        name_idx = idx
    elif any(
        k in col_lower
        for k in [
            '緯度経度',
            '経度緯度',
            '座標',
            'latlon',
            'location',
            'coords',
        ]
    ):
      latlon_idx = idx
    elif any(k in col_lower for k in ['緯度', 'lat', 'latitude']):
      lat_idx = idx
    elif any(k in col_lower for k in ['経度', 'lon', 'lng', 'longitude']):
      lon_idx = idx

  spots = []
  for row_num, row_data in enumerate(df.itertuples(index=False), start=2):
    row = [str(val) if pd.notna(val) else '' for val in row_data]
    if not row or all(c.strip() == '' for c in row):
      continue
    name = (
        row[name_idx].strip()
        if (name_idx != -1 and name_idx < len(row))
        else f'地点_{row_num-1}'
    )
    lat, lon = None, None

    if latlon_idx != -1 and latlon_idx < len(row):
      lat, lon = parse_lat_lon_pair(row[latlon_idx])
    elif (
        lat_idx != -1
        and lon_idx != -1
        and lat_idx < len(row)
        and lon_idx < len(row)
    ):
      lat, lon = parse_lat_lon_pair(row[lat_idx], row[lon_idx])
    else:
      for cell in row:
        t_lat, t_lon = parse_lat_lon_pair(cell)
        if t_lat is not None and t_lon is not None:
          lat, lon = t_lat, t_lon
          break

    if lat is not None and lon is not None:
      spots.append({'lon': lon, 'lat': lat, 'name': name})

  return spots


def parse_bytes_content(content, filename_hint=''):
  filename_hint = filename_hint.lower()
  is_xlsx = content.startswith(b'PK\x03\x04') or filename_hint.endswith(
      '.xlsx'
  )
  is_xls = content.startswith(b'\xd0\xcf\x11\xe0') or filename_hint.endswith(
      '.xls'
  )

  if is_xlsx or is_xls:
    df = pd.read_excel(io.BytesIO(content))
    return parse_dataframe_to_spots(df)

  text = None
  for enc in ['utf-8-sig', 'cp932', 'utf-8', 'shift_jis']:
    try:
      text = content.decode(enc)
      break
    except Exception:
      continue

  if not text:
    st.error('ファイルの文字コードを判別できませんでした。')
    return []

  stripped_text = text.strip()
  if (
      (stripped_text.startswith('{') or stripped_text.startswith('['))
      or filename_hint.endswith('.geojson')
      or filename_hint.endswith('.json')
  ):
    try:
      geojson_data = json.loads(text)
      spots = []

      def extract_geom(geom, props):
        if not geom:
          return
        g_type = geom.get('type')
        coords = geom.get('coordinates')
        if not coords:
          return
        name = props.get(
            'name', props.get('title', props.get('管理番号', '未命名の地点'))
        )

        if g_type == 'Point':
          spots.append(
              {'lon': float(coords[0]), 'lat': float(coords[1]), 'name': name}
          )
        elif g_type in ['MultiPoint', 'LineString']:
          for i, pt in enumerate(coords):
            spots.append({
                'lon': float(pt[0]),
                'lat': float(pt[1]),
                'name': f'{name}_{i}',
            })
        elif g_type in ['Polygon']:
          for i, pt in enumerate(coords[0]):
            spots.append({
                'lon': float(pt[0]),
                'lat': float(pt[1]),
                'name': f'{name}_{i}',
            })

      if geojson_data.get('type') == 'FeatureCollection':
        for feature in geojson_data.get('features', []):
          extract_geom(feature.get('geometry'), feature.get('properties', {}))
      elif geojson_data.get('type') == 'Feature':
        extract_geom(
            geojson_data.get('geometry'), feature.get('properties', {})
        )
      else:
        extract_geom(geojson_data, {})

      seen = set()
      unique_spots = []
      for s in spots:
        coord_key = (s['lon'], s['lat'])
        if coord_key not in seen:
          seen.add(coord_key)
          unique_spots.append(s)
      return unique_spots
    except Exception:
      pass

  reader = csv.reader(io.StringIO(text))
  header = [h.strip().replace('\ufeff', '') for h in next(reader, [])]
  rows = list(reader)
  if header and rows:
    df = pd.DataFrame(rows, columns=header[: len(rows[0])])
    return parse_dataframe_to_spots(df)

  return []


# ==========================================
# 2. OSRM API ＆ TSP計算部
# ==========================================


def get_osrm_matrix(spots):
  coords_str = ';'.join(f"{s['lon']},{s['lat']}" for s in spots)
  url = f'https://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=duration'
  req = urllib.request.Request(
      url, headers={'User-Agent': 'TSP-OSRM-App-Agent'}
  )
  with urllib.request.urlopen(req, timeout=15) as response:
    data = json.loads(response.read().decode())
  if 'durations' not in data:
    raise Exception('OSRMから移動時間のデータを取得できませんでした。')
  return data['durations']


def get_osrm_route_geometry(ordered_spots):
  coords_str = ';'.join(f"{s['lon']},{s['lat']}" for s in ordered_spots)
  url = f'https://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson'
  req = urllib.request.Request(
      url, headers={'User-Agent': 'TSP-OSRM-App-Agent'}
  )
  with urllib.request.urlopen(req, timeout=15) as response:
    data = json.loads(response.read().decode())
  if 'routes' not in data or len(data['routes']) == 0:
    raise Exception('OSRMから走行ルートの形状を取得できませんでした。')
  return data['routes'][0]['geometry']['coordinates']


def solve_tsp(duration_matrix, is_round_trip=True):
  n = len(duration_matrix)
  if is_round_trip:
    if n <= 2:
      return list(range(n)) + [0]
    unvisited = set(range(1, n))
    curr = 0
    tour = [0]
    while unvisited:
      nearest = min(unvisited, key=lambda idx: duration_matrix[curr][idx])
      tour.append(nearest)
      unvisited.remove(nearest)
      curr = nearest
    tour.append(0)

    improved = True
    while improved:
      improved = False
      for i in range(1, n - 1):
        for j in range(i + 1, n):
          current_cost = (
              duration_matrix[tour[i - 1]][tour[i]]
              + duration_matrix[tour[j]][tour[j + 1]]
          )
          new_cost = (
              duration_matrix[tour[i - 1]][tour[j]]
              + duration_matrix[tour[i]][tour[j + 1]]
          )
          if new_cost < current_cost - 1e-3:
            tour[i : j + 1] = reversed(tour[i : j + 1])
            improved = True
            break
        if improved:
          break
    return tour
  else:
    if n <= 2:
      return list(range(n))
    unvisited = set(range(1, n - 1))
    curr = 0
    tour = [0]
    while unvisited:
      nearest = min(unvisited, key=lambda idx: duration_matrix[curr][idx])
      tour.append(nearest)
      unvisited.remove(nearest)
      curr = nearest
    tour.append(n - 1)

    improved = True
    while improved:
      improved = False
      for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
          current_cost = (
              duration_matrix[tour[i - 1]][tour[i]]
              + duration_matrix[tour[j]][tour[j + 1]]
          )
          new_cost = (
              duration_matrix[tour[i - 1]][tour[j]]
              + duration_matrix[tour[i]][tour[j + 1]]
          )
          if new_cost < current_cost - 1e-3:
            tour[i : j + 1] = reversed(tour[i : j + 1])
            improved = True
            break
        if improved:
          break
    return tour


def generate_google_maps_urls(ordered_spots, max_waypoints=9):
  chunk_size = max_waypoints + 1
  urls_info = []
  total_pts = len(ordered_spots)
  idx = 0
  part = 1

  while idx < total_pts - 1:
    end_idx = min(idx + chunk_size, total_pts - 1)
    sub_spots = ordered_spots[idx : end_idx + 1]
    path_coords = '/'.join(f"{s['lat']},{s['lon']}" for s in sub_spots)
    url = f'https://www.google.com/maps/dir/{path_coords}'

    urls_info.append({
        'part': part,
        'count': len(sub_spots) - 1,
        'start': sub_spots[0]['name'],
        'end': sub_spots[-1]['name'],
        'url': url,
    })
    idx = end_idx
    part += 1

  return urls_info


# ==========================================
# 3. Streamlit WEB GUI 部（スマホ対応デザイン）
# ==========================================

st.set_page_config(
    page_title='現場向け TSPルート作成', page_icon='🚚', layout='wide'
)

# 📱 スマホ画面表示用のカスタムCSS（パディング削減・ボタン拡大）
st.markdown(
    """
<style>
    /* 画面幅いっぱいにコンテンツを広げる */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    /* ボタンを太く・押しやすく（おっちゃんの指用） */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        padding-top: 0.6rem;
        padding-bottom: 0.6rem;
        font-weight: bold;
    }
    /* タイトルの見やすさ調整 */
    h1 {
        font-size: 1.5rem !important;
        padding-bottom: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title('🚚 現場向け 最短ルート作成 ＆ Google Maps生成')

# --- Widget State の初期化 ---
if 'input_start' not in st.session_state:
  st.session_state['input_start'] = '35.9655, 140.2942'
if 'input_end' not in st.session_state:
  st.session_state['input_end'] = ''
if 'map_center' not in st.session_state:
  st.session_state['map_center'] = (36.0, 140.0)
if 'map_zoom' not in st.session_state:
  st.session_state['map_zoom'] = 10


# --- コールバック関数群 ---
def set_start_from_center():
  if 'last_dragged_center' in st.session_state:
    st.session_state['map_center'] = st.session_state['last_dragged_center']
  if 'last_dragged_zoom' in st.session_state:
    st.session_state['map_zoom'] = st.session_state['last_dragged_zoom']

  c_lat, c_lon = st.session_state['map_center']
  st.session_state['input_start'] = f'{c_lat:.6f}, {c_lon:.6f}'


def set_end_from_center():
  if 'last_dragged_center' in st.session_state:
    st.session_state['map_center'] = st.session_state['last_dragged_center']
  if 'last_dragged_zoom' in st.session_state:
    st.session_state['map_zoom'] = st.session_state['last_dragged_zoom']

  c_lat, c_lon = st.session_state['map_center']
  st.session_state['input_end'] = f'{c_lat:.6f}, {c_lon:.6f}'


def search_and_set_start():
  q = st.session_state.get('search_query_box', '')
  if q:
    g_lat, g_lon = search_address_to_latlon(q)
    if g_lat and g_lon:
      st.session_state['input_start'] = f'{g_lat:.6f}, {g_lon:.6f}'
      st.session_state['map_center'] = (g_lat, g_lon)
      st.session_state['map_zoom'] = 16
      st.session_state['last_dragged_center'] = (g_lat, g_lon)
      st.session_state['last_dragged_zoom'] = 16


def search_and_set_end():
  q = st.session_state.get('search_query_box', '')
  if q:
    g_lat, g_lon = search_address_to_latlon(q)
    if g_lat and g_lon:
      st.session_state['input_end'] = f'{g_lat:.6f}, {g_lon:.6f}'
      st.session_state['map_center'] = (g_lat, g_lon)
      st.session_state['map_zoom'] = 16
      st.session_state['last_dragged_center'] = (g_lat, g_lon)
      st.session_state['last_dragged_zoom'] = 16


# --- 1. 読み込み方法の選択 ---
load_type = st.radio(
    '① データの読み込み方法',
    ['📁 PCファイルアップロード', '☁️ Google Drive リンク'],
    horizontal=True,
)

file_bytes = None
file_hint_name = ''

if load_type == '📁 PCファイルアップロード':
  uploaded_file = st.file_uploader(
      'Excel / CSV / GeoJSON を選択',
      type=['xlsx', 'xls', 'csv', 'geojson', 'json'],
  )
  if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hint_name = uploaded_file.name
else:
    gdrive_url = st.text_input(
        'Google Drive共有URL',
        placeholder='https://drive.google.com/file/d/...',
    )
    if gdrive_url:
      file_id = extract_gdrive_file_id(gdrive_url)
      if file_id:
        try:
          file_bytes = download_from_gdrive(file_id)
          st.success('☁️ 取得成功！')
        except Exception as e:
          st.error(f'ダウンロード失敗: {e}')

preview_spots = []
if file_bytes:
  try:
    preview_spots = parse_bytes_content(file_bytes, file_hint_name)
  except Exception:
    pass


# --- 2. 出発地 ＆ 終点地（ゴール）の設定エリア ---
st.markdown('---')
st.subheader('📍 出発地・終点地（ゴール）の設定')

# A. 住所・施設名検索（スマホで潰れないように2列配置）
st.text_input(
    '🔍 住所・施設名から検索',
    placeholder='例: 東京都千代田区丸の内1-9-1 または 東京タワー',
    key='search_query_box',
)
col_a1, col_a2 = st.columns(2)
with col_a1:
  st.button(
      '📍 出発地に検索セット',
      on_click=search_and_set_start,
      use_container_width=True,
  )
with col_a2:
  st.button(
      '🏁 終点地に検索セット',
      on_click=search_and_set_end,
      use_container_width=True,
  )

st.write('')

# B. 出発地入力 ＆ 地図中心セットボタン
st.text_input('📍 出発地 (lat, lon)', key='input_start')
st.button(
    '📍 地図中心（✚）を出発地にセット',
    on_click=set_start_from_center,
    use_container_width=True,
)

st.write('')

# C. 終点地入力 ＆ 地図中心セットボタン
st.text_input(
    '🏁 終点地 (lat, lon ※空欄で戻る)', key='input_end'
)
st.button(
    '🏁 地図中心（✚）を終点地にセット',
    on_click=set_end_from_center,
    use_container_width=True,
)

st.caption(
    '💡 使い方: 上で住所検索するか、地図を指で動かして中央の赤十字（✚）を合わせ、セットボタンを押してください。'
)

# --- 地図描画 ---
s_lat, s_lon = parse_lat_lon_pair(st.session_state['input_start'])
e_lat, e_lon = parse_lat_lon_pair(st.session_state['input_end'])

m_input = folium.Map(
    location=st.session_state['map_center'],
    zoom_start=st.session_state['map_zoom'],
)

# 中央固定の赤十字（✚）オーバーレイ
crosshair_html = """
<div style="
    position: absolute;
    top: 50%;
    left: 50%;
    width: 30px;
    height: 30px;
    margin-top: -15px;
    margin-left: -15px;
    z-index: 9999;
    pointer-events: none;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    color: #d32f2f;
    font-weight: bold;
    text-shadow: 0 0 3px #ffffff, 0 0 3px #ffffff, 0 0 5px #ffffff;
">✚</div>
"""
m_input.get_root().html.add_child(folium.Element(crosshair_html))

# プレビュー表示
for s in preview_spots:
  folium.Marker(
      [s['lat'], s['lon']],
      popup=s['name'],
      tooltip=s['name'],
      icon=folium.Icon(color='blue', icon='info-sign'),
  ).add_to(m_input)

if s_lat and s_lon:
  folium.Marker(
      [s_lat, s_lon],
      popup='★ 現在の出発地',
      tooltip='★ 現在の出発地',
      icon=folium.Icon(color='red', icon='play'),
  ).add_to(m_input)

if e_lat and e_lon:
  folium.Marker(
      [e_lat, e_lon],
      popup='🏁 現在の終点地',
      tooltip='🏁 現在の終点地',
      icon=folium.Icon(color='green', icon='flag'),
  ).add_to(m_input)

# スマホ向けに地図の高さ(height)を 380px に最適化
map_output = st_folium(
    m_input,
    key='select_map',
    use_container_width=True,
    height=380,
    returned_objects=['center', 'zoom'],
)

if map_output and 'center' in map_output and map_output['center']:
  c_lat = map_output['center']['lat']
  c_lon = map_output['center']['lng']
  st.session_state['last_dragged_center'] = (c_lat, c_lon)

if map_output and 'zoom' in map_output and map_output['zoom']:
  st.session_state['last_dragged_zoom'] = map_output['zoom']


# --- 3. 計算実行ボタン ---
st.markdown('---')
if st.button(
    '🚀 最短ルート計算 & Google Maps URL作成',
    type='primary',
    use_container_width=True,
):
  if not file_bytes:
    st.warning('⚠️ ファイルの指定が必要です。')
  else:
    with st.spinner('計算中...'):
      try:
        raw_spots = parse_bytes_content(file_bytes, file_hint_name)
        if not raw_spots:
          st.error('有効なスポットがありません。')
          st.stop()

        s_lat, s_lon = parse_lat_lon_pair(st.session_state['input_start'])
        if s_lat is None or s_lon is None:
          st.error('📍 出発地を指定してください。')
          st.stop()

        start_spot = {'lon': s_lon, 'lat': s_lat, 'name': '指定された出発地'}
        e_lat, e_lon = parse_lat_lon_pair(st.session_state['input_end'])

        if e_lat is None or e_lon is None:
          is_round_trip = True
          calc_spots = [start_spot] + raw_spots
        else:
          is_round_trip = False
          end_spot = {'lon': e_lon, 'lat': e_lat, 'name': '指定された終点地'}
          calc_spots = [start_spot] + raw_spots + [end_spot]

        duration_matrix = get_osrm_matrix(calc_spots)
        tour = solve_tsp(duration_matrix, is_round_trip=is_round_trip)
        ordered_spots = [calc_spots[idx] for idx in tour]
        detailed_route = get_osrm_route_geometry(ordered_spots)
        urls_info = generate_google_maps_urls(ordered_spots, max_waypoints=9)

        st.session_state['result'] = {
            'is_round_trip': is_round_trip,
            'tour': tour,
            'calc_spots': calc_spots,
            'ordered_spots': ordered_spots,
            'detailed_route': detailed_route,
            'urls_info': urls_info,
        }
        st.success(f'🎉 計算完了！（全 {len(raw_spots)} スポット）')

      except Exception as e:
        st.error(f'計算エラー: {e}')

# --- 4. 結果表示 ---
if 'result' in st.session_state:
  res = st.session_state['result']
  mode_text = '【周回】' if res['is_round_trip'] else '【片道】'

  st.markdown('---')
  st.subheader(f'🔗 Google Maps URL {mode_text}')

  for item in res['urls_info']:
    st.markdown(f"**Part {item['part']}**: `{item['start']}` ➔ `{item['end']}`")
    st.link_button(
        f"🌐 Part {item['part']} のナビを開く",
        item['url'],
        use_container_width=True,
    )
    st.write('')

  st.markdown('---')
  st.subheader('🗺️ 最適化ルートマップ')

  avg_lat = sum(s['lat'] for s in res['ordered_spots']) / len(
      res['ordered_spots']
  )
  avg_lon = sum(s['lon'] for s in res['ordered_spots']) / len(
      res['ordered_spots']
  )

  m = folium.Map(location=[avg_lat, avg_lon], zoom_start=11)

  route_latlons = [(lat, lon) for lon, lat in res['detailed_route']]
  folium.PolyLine(
      route_latlons, color='#d32f2f', weight=5, opacity=0.8
  ).add_to(m)

  total_len = len(res['tour'])
  for order, idx in enumerate(res['tour']):
    if res['is_round_trip'] and order == total_len - 1:
      continue

    spot = res['calc_spots'][idx]

    if order == 0:
      txt = 'START / GOAL' if res['is_round_trip'] else 'START'
      folium.Marker(
          [spot['lat'], spot['lon']],
          popup=txt,
          tooltip=txt,
          icon=folium.Icon(color='red', icon='play'),
      ).add_to(m)
    elif not res['is_round_trip'] and order == total_len - 1:
      folium.Marker(
          [spot['lat'], spot['lon']],
          popup=f"GOAL: {spot['name']}",
          tooltip=f"GOAL: {spot['name']}",
          icon=folium.Icon(color='green', icon='flag'),
      ).add_to(m)
    else:
      folium.Marker(
          [spot['lat'], spot['lon']],
          popup=f"[{order}] {spot['name']}",
          tooltip=f"[{order}] {spot['name']}",
          icon=folium.Icon(color='blue', icon='info-sign'),
      ).add_to(m)

  st_folium(m, use_container_width=True, height=450)