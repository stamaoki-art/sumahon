import csv
import io
import json
import re
import urllib.parse
import urllib.request
import folium
from folium.plugins import LocateControl
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium

# ==========================================
# 1. Web API 住所・施設検索部 ＆ 距離計算部
# ==========================================


def search_address_to_latlon(address_text):
  """住所・施設名から緯度経度(lat, lon)を取得"""
  if not address_text or not address_text.strip():
    return None, None

  query = address_text.strip()
  encoded_query = urllib.parse.quote(query)

  try:
    gsi_url = f'https://msearch.gsi.go.jp/address-search/AddressSearch?q={encoded_query}'
    req = urllib.request.Request(
        gsi_url, headers={'User-Agent': 'TSP-Field-App'}
    )
    with urllib.request.urlopen(req, timeout=5) as response:
      data = json.loads(response.read().decode())
      if data and len(data) > 0 and 'geometry' in data[0]:
        coords = data[0]['geometry']['coordinates']
        return round(float(coords[1]), 5), round(float(coords[0]), 5)
  except Exception:
    pass

  try:
    osm_url = f'https://nominatim.openstreetmap.org/search?q={encoded_query}&format=json&limit=1'
    req = urllib.request.Request(
        osm_url, headers={'User-Agent': 'TSP-Field-App'}
    )
    with urllib.request.urlopen(req, timeout=5) as response:
      data = json.loads(response.read().decode())
      if data and len(data) > 0:
        return round(float(data[0]['lat']), 5), round(float(data[0]['lon']), 5)
  except Exception:
    pass

  return None, None


def parse_lat_lon_pair(val1, val2=None):
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
      return round(n2, 5), round(n1, 5)
    elif 20 <= n1 <= 50 and 120 <= n2 <= 155:
      return round(n1, 5), round(n2, 5)
    else:
      return round(n1, 5), round(n2, 5)
  except Exception:
    return None, None


def deduplicate_spots(spots, start_coord=None):
  """完全一致する座標を検知し、重複を除外する。

  start_coord が指定された場合、出発地と同一座標のスポットも除外する。
  戻り値: (unique_spots, duplicates)
  """
  seen = set()
  if start_coord is not None:
    seen.add((round(start_coord[0], 5), round(start_coord[1], 5)))

  unique_spots = []
  duplicates = []
  for s in spots:
    coord_key = (round(s['lat'], 5), round(s['lon'], 5))
    if coord_key in seen:
      duplicates.append(s)
    else:
      seen.add(coord_key)
      unique_spots.append(s)

  return unique_spots, duplicates


def calculate_haversine_matrix(spots):
  lats = np.radians([s['lat'] for s in spots])
  lons = np.radians([s['lon'] for s in spots])
  dlat = lats[:, np.newaxis] - lats[np.newaxis, :]
  dlon = lons[:, np.newaxis] - lons[np.newaxis, :]
  a = (
      np.sin(dlat / 2.0) ** 2
      + np.cos(lats[:, np.newaxis])
      * np.cos(lats[np.newaxis, :])
      * np.sin(dlon / 2.0) ** 2
  )
  c = 2 * np.arcsin(np.sqrt(a))
  r = 6371000.0
  return c * r


def get_osrm_matrix_chunked(spots, chunk_size=15):
  n = len(spots)
  matrix = np.zeros((n, n))
  chunks = [
      list(range(i, min(i + chunk_size, n))) for i in range(0, n, chunk_size)
  ]
  try:
    for src_chunk in chunks:
      for dst_chunk in chunks:
        combined_indices = list(dict.fromkeys(src_chunk + dst_chunk))
        src_local = [combined_indices.index(idx) for idx in src_chunk]
        dst_local = [combined_indices.index(idx) for idx in dst_chunk]
        coords_str = ';'.join(
            f"{spots[idx]['lon']:.5f},{spots[idx]['lat']:.5f}"
            for idx in combined_indices
        )
        src_str = ';'.join(map(str, src_local))
        dst_str = ';'.join(map(str, dst_local))
        url = f'https://router.project-osrm.org/table/v1/driving/{coords_str}?sources={src_str}&destinations={dst_str}&annotations=duration'
        req = urllib.request.Request(
            url, headers={'User-Agent': 'TSP-OSRM-App-Agent'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
          data = json.loads(response.read().decode())
          durations = data['durations']
          for s_i, src_idx in enumerate(src_chunk):
            for d_j, dst_idx in enumerate(dst_chunk):
              matrix[src_idx, dst_idx] = durations[s_i][d_j]
    return matrix
  except Exception:
    return calculate_haversine_matrix(spots)


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
        for k in ['管理番号', '名称', '名前', '設備', 'id', 'スポット']
    ):
      if name_idx == -1:
        name_idx = idx
    elif any(
        k in col_lower
        for k in ['緯度経度', '経度緯度', '座標', 'latlon', 'location', 'coords']
    ):
      latlon_idx = idx
    elif any(k in col_lower for k in ['緯度', 'lat', 'latitude']):
      lat_idx = idx
    elif any(k in col_lower for k in ['経度', 'lon', 'lng', 'longitude']):
      lon_idx = idx

  spots = []
  for row_num, row_data in enumerate(df.itertuples(index=False), start=2):
    row_dict = dict(zip(header, row_data))
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
    if (
        (lat is None or lon is None)
        and lat_idx != -1
        and lon_idx != -1
        and lat_idx < len(row)
        and lon_idx < len(row)
    ):
      lat, lon = parse_lat_lon_pair(row[lat_idx], row[lon_idx])
    if lat is None or lon is None:
      for cell in row:
        t_lat, t_lon = parse_lat_lon_pair(cell)
        if t_lat is not None and t_lon is not None:
          lat, lon = t_lat, t_lon
          break

    if lat is not None and lon is not None:
      spots.append({
          'lon': round(lon, 5),
          'lat': round(lat, 5),
          'name': name,
          'raw_dict': row_dict,
      })
  return spots


def parse_bytes_content(content, filename_hint=''):
  filename_hint = filename_hint.lower()
  is_xlsx = content.startswith(b'PK\x03\x04') or filename_hint.endswith('.xlsx')
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
          spots.append({
              'lon': round(float(coords[0]), 5),
              'lat': round(float(coords[1]), 5),
              'name': name,
              'raw_dict': props,
          })
        elif g_type in ['MultiPoint', 'LineString']:
          for i, pt in enumerate(coords):
            spots.append({
                'lon': round(float(pt[0]), 5),
                'lat': round(float(pt[1]), 5),
                'name': f'{name}_{i}',
                'raw_dict': props,
            })
        elif g_type in ['Polygon']:
          for i, pt in enumerate(coords[0]):
            spots.append({
                'lon': round(float(pt[0]), 5),
                'lat': round(float(pt[1]), 5),
                'name': f'{name}_{i}',
                'raw_dict': props,
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


def get_osrm_route_geometry_chunked(ordered_spots, max_chunk=25):
  total_spots = len(ordered_spots)
  all_coordinates = []
  idx = 0
  while idx < total_spots - 1:
    end_idx = min(idx + max_chunk, total_spots - 1)
    sub_spots = ordered_spots[idx : end_idx + 1]
    coords_str = ';'.join(f"{s['lon']:.5f},{s['lat']:.5f}" for s in sub_spots)
    url = f'https://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson&continue_straight=false'
    req = urllib.request.Request(
        url, headers={'User-Agent': 'TSP-OSRM-App-Agent'}
    )
    try:
      with urllib.request.urlopen(req, timeout=15) as response:
        data = json.loads(response.read().decode())
      if 'routes' in data and len(data['routes']) > 0:
        sub_coords = data['routes'][0]['geometry']['coordinates']
        if all_coordinates:
          all_coordinates.extend(sub_coords[1:])
        else:
          all_coordinates.extend(sub_coords)
    except Exception:
      pass
    idx = end_idx
  return all_coordinates


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
    limit = 10000 if n < 150 else 1000
    count = 0
    while improved and count < limit:
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
            count += 1
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
    limit = 10000 if n < 150 else 1000
    count = 0
    while improved and count < limit:
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
            count += 1
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

    path_coords = '/'.join(f"{s['lat']:.5f},{s['lon']:.5f}" for s in sub_spots)
    url = f'https://www.google.com/maps/dir/{path_coords}'

    next_spots_coords = '/'.join(
        f"{s['lat']:.5f},{s['lon']:.5f}" for s in sub_spots[1:]
    )
    current_loc_url = f'https://www.google.com/maps/dir//{next_spots_coords}'

    urls_info.append({
        'part': part,
        'count': len(sub_spots) - 1,
        'start': sub_spots[0]['name'],
        'end': sub_spots[-1]['name'],
        'url': url,
        'current_loc_url': current_loc_url,
    })
    idx = end_idx
    part += 1

  return urls_info


def create_export_dataframe(res):
  tour = res['tour']
  calc_spots = res['calc_spots']
  is_round_trip = res['is_round_trip']
  total_len = len(tour)

  export_rows = []
  for order, idx in enumerate(tour):
    if is_round_trip and order == total_len - 1:
      continue

    spot = calc_spots[idx]
    raw = spot.get('raw_dict', {})

    if order == 0:
      visit_label = '出発地'
    elif not is_round_trip and order == total_len - 1:
      visit_label = '終点地'
    else:
      visit_label = f'立ち寄り_{order}'

    row_data = {
        '訪問順': order,
        '区分': visit_label,
        '名称/識別ID': spot['name'],
        '緯度': round(spot['lat'], 5),
        '経度': round(spot['lon'], 5),
    }

    for k, v in raw.items():
      if k not in row_data:
        row_data[k] = v

    export_rows.append(row_data)

  return pd.DataFrame(export_rows)


# ==========================================
# 3. HTMLスタンドアローン化（消去防止）
# ==========================================
def create_result_map(res):
  avg_lat = sum(s['lat'] for s in res['ordered_spots']) / len(
      res['ordered_spots']
  )
  avg_lon = sum(s['lon'] for s in res['ordered_spots']) / len(
      res['ordered_spots']
  )

  m = folium.Map(location=[avg_lat, avg_lon], zoom_start=11, tiles=None)

  folium.TileLayer('OpenStreetMap', name='標準地図').add_to(m)
  folium.TileLayer(
      tiles='https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg',
      attr='国土地理院',
      name='📡 航空写真',
      max_zoom=18,
  ).add_to(m)
  folium.LayerControl(position='topright').add_to(m)

  LocateControl(
      auto_start=False,
      flyTo=True,
      keepCurrentZoomLevel=False,
      strings={'title': 'GPS現在地を表示'},
  ).add_to(m)

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

  return m


def generate_standalone_html(res, m):
  mode_text = '【周回】' if res['is_round_trip'] else '【片道】'

  links_html = f"""
  <div style="padding: 15px; font-family: sans-serif; background-color: #f8f9fa; border-bottom: 2px solid #ccc; max-height: 40vh; overflow-y: auto;">
      <h3 style="margin-top: 0; font-size: 18px;">🔗 Google Maps ナビURL {mode_text}</h3>
      <p style="font-size: 13px; color: #d32f2f; font-weight: bold; margin-bottom: 10px;">
          ※この画面はオフラインでも機能し、ブラウザがリロードされても消えません！このままナビにお使いください。
      </p>
      <div style="display: flex; flex-direction: column; gap: 10px;">
  """

  for item in res['urls_info']:
    links_html += f"""
      <div style="background: white; padding: 10px; border-radius: 8px; border: 1px solid #ddd;">
          <div style="font-weight: bold; margin-bottom: 8px; font-size: 14px;">
              Part {item['part']}: {item['start']} ➔ {item['end']} ({item['count']}区間)
          </div>
          <div style="display: flex; gap: 10px;">
              <a href="{item['url']}" target="_blank" style="flex: 1; text-align: center; text-decoration: none; background-color: #e0e0e0; color: #333; padding: 10px; border-radius: 5px; font-weight: bold; font-size: 14px;">🚗 通常ルート</a>
              <a href="{item['current_loc_url']}" target="_blank" style="flex: 1; text-align: center; text-decoration: none; background-color: #0288d1; color: white; padding: 10px; border-radius: 5px; font-weight: bold; font-size: 14px;">📍 現在地から開始</a>
          </div>
      </div>
      """
  links_html += """
      </div>
  </div>
  """

  map_html = m.get_root().render()

  final_html = map_html.replace(
      '<body>',
      f'<body style="margin:0; padding:0; display:flex; flex-direction:column; height:100vh; overflow:hidden;">\n{links_html}\n<div style="flex-grow:1; position:relative;">',
      1,
  )
  final_html = final_html.replace('</body>', '</div></body>')

  return final_html


# ==========================================
# 4. Streamlit WEB GUI 部
# ==========================================

st.set_page_config(
    page_title='現場向け TSPルート作成', page_icon='🚚', layout='wide'
)

st.markdown(
    """
<style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    div.stButton > button, div.stDownloadButton > button {
        width: 100%;
        border-radius: 8px;
        padding-top: 0.6rem;
        padding-bottom: 0.6rem;
        font-weight: bold;
    }
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
  st.session_state['input_start'] = '35.96550, 140.29420'
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
  st.session_state['input_start'] = f'{c_lat:.5f}, {c_lon:.5f}'


def set_end_from_center():
  if 'last_dragged_center' in st.session_state:
    st.session_state['map_center'] = st.session_state['last_dragged_center']
  if 'last_dragged_zoom' in st.session_state:
    st.session_state['map_zoom'] = st.session_state['last_dragged_zoom']
  c_lat, c_lon = st.session_state['map_center']
  st.session_state['input_end'] = f'{c_lat:.5f}, {c_lon:.5f}'


def search_and_set_start():
  q = st.session_state.get('search_query_box', '')
  if q:
    g_lat, g_lon = search_address_to_latlon(q)
    if g_lat and g_lon:
      st.session_state['input_start'] = f'{g_lat:.5f}, {g_lon:.5f}'
      st.session_state['map_center'] = (g_lat, g_lon)
      st.session_state['map_zoom'] = 16
      st.session_state['last_dragged_center'] = (g_lat, g_lon)
      st.session_state['last_dragged_zoom'] = 16


def search_and_set_end():
  q = st.session_state.get('search_query_box', '')
  if q:
    g_lat, g_lon = search_address_to_latlon(q)
    if g_lat and g_lon:
      st.session_state['input_end'] = f'{g_lat:.5f}, {g_lon:.5f}'
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
    raw_preview = parse_bytes_content(file_bytes, file_hint_name)
    preview_spots, preview_dups = deduplicate_spots(raw_preview)
    if preview_dups:
      st.warning(
          f'⚠️ ファイル内に完全一致する重複座標が {len(preview_dups)}'
          ' 件見つかりました。ルート計算時に自動で除外されます。'
      )
  except Exception:
    pass


# --- 2. 出発地 ＆ 終点地（ゴール）の設定エリア ---
st.markdown('---')
st.subheader('📍 出発地・終点地（ゴール）の設定')

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

st.text_input('📍 出発地 (lat, lon)', key='input_start')
st.button(
    '📍 地図中心（✚）を出発地にセット',
    on_click=set_start_from_center,
    use_container_width=True,
)

st.write('')

st.text_input('🏁 終点地 (lat, lon ※空欄で戻る)', key='input_end')
st.button(
    '🏁 終点地に（✚）をセット',
    on_click=set_end_from_center,
    use_container_width=True,
)

st.caption(
    '💡 使い方:'
    ' 地図左上の「🎯（GPSマーク）」を押すと現在地に自動移動します。右上のボタンで「航空写真」にも切り替えられます！'
)

# --- 地図描画 ---
s_lat, s_lon = parse_lat_lon_pair(st.session_state['input_start'])
e_lat, e_lon = parse_lat_lon_pair(st.session_state['input_end'])

m_input = folium.Map(
    location=st.session_state['map_center'],
    zoom_start=st.session_state['map_zoom'],
    tiles=None,
)

folium.TileLayer('OpenStreetMap', name='標準地図').add_to(m_input)
folium.TileLayer(
    tiles='https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg',
    attr='国土地理院',
    name='📡 航空写真',
    max_zoom=18,
).add_to(m_input)
folium.LayerControl(position='topright').add_to(m_input)

LocateControl(
    auto_start=False,
    flyTo=True,
    keepCurrentZoomLevel=False,
    strings={'title': 'GPS現在地を表示'},
).add_to(m_input)

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
  st.session_state['last_dragged_center'] = (round(c_lat, 5), round(c_lon, 5))

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
    with st.spinner('切り返し・実道路網を考慮して最適ルートを計算中...'):
      try:
        raw_spots = parse_bytes_content(file_bytes, file_hint_name)
        if not raw_spots:
          st.error('有効なスポットがありません。')
          st.stop()

        s_lat, s_lon = parse_lat_lon_pair(st.session_state['input_start'])
        if s_lat is None or s_lon is None:
          st.error('📍 出発地を指定してください。')
          st.stop()

        # ★【重複座標自動検知＆除外システム】出発地との被りも含めて完全一致を排除！
        unique_spots, duplicates = deduplicate_spots(
            raw_spots, start_coord=(s_lat, s_lon)
        )

        if duplicates:
          dup_names = [f"「{d['name']}」" for d in duplicates[:5]]
          more_txt = (
              f' 他 {len(duplicates)-5} 件' if len(duplicates) > 5 else ''
          )
          st.warning(
              f'⚠️ Google Mapsのエラーを防ぐため、完全一致する重複座標 {len(duplicates)}'
              f' 件を除外してルート計算しました。\n\n（除外対象:'
              f' {", ".join(dup_names)}{more_txt}）'
          )

        if not unique_spots:
          st.error(
              '有効な立ち寄り先がありません（すべて出発地と同じか重複しています）。'
          )
          st.stop()

        start_spot = {'lon': s_lon, 'lat': s_lat, 'name': '指定された出発地'}
        e_lat, e_lon = parse_lat_lon_pair(st.session_state['input_end'])

        if e_lat is None or e_lon is None:
          is_round_trip = True
          calc_spots = [start_spot] + unique_spots
        else:
          is_round_trip = False
          end_spot = {'lon': e_lon, 'lat': e_lat, 'name': '指定された終点地'}
          calc_spots = [start_spot] + unique_spots + [end_spot]

        dist_matrix = get_osrm_matrix_chunked(calc_spots, chunk_size=15)
        tour = solve_tsp(dist_matrix, is_round_trip=is_round_trip)
        ordered_spots = [calc_spots[idx] for idx in tour]

        detailed_route = get_osrm_route_geometry_chunked(
            ordered_spots, max_chunk=25
        )
        urls_info = generate_google_maps_urls(ordered_spots, max_waypoints=9)

        st.session_state['result'] = {
            'is_round_trip': is_round_trip,
            'tour': tour,
            'calc_spots': calc_spots,
            'ordered_spots': ordered_spots,
            'detailed_route': detailed_route,
            'urls_info': urls_info,
            'duplicates': duplicates,
        }

        res_map = create_result_map(st.session_state['result'])
        st.session_state['result_map'] = res_map
        st.session_state['standalone_html'] = generate_standalone_html(
            st.session_state['result'], res_map
        )

        st.success(
            f'🎉 計算完了！（立ち寄り先: {len(unique_spots)} 件'
            + (f' / 重複除外: {len(duplicates)} 件' if duplicates else '')
            + '）'
        )

      except Exception as e:
        st.error(f'計算エラー: {e}')

# --- 4. 結果表示 ＆ ダウンロードエリア ---
if 'result' in st.session_state:
  res = st.session_state['result']
  mode_text = '【周回】' if res['is_round_trip'] else '【片道】'

  # A. データ吹き飛び防止HTMLボタン
  if 'standalone_html' in st.session_state:
    st.markdown('---')
    st.subheader('🛡️ データ吹き飛び防止（超重要！）')
    st.error(
        'スマホでは時間が経つと勝手にリロードされて結果が消えてしまいます！以下のボタンから**「消えない別画面」**を開いて使ってください！'
    )

    col_jump1, col_jump2 = st.columns(2)
    with col_jump1:
      html_encoded = urllib.parse.quote(st.session_state['standalone_html'])
      components.html(
          f"""
          <button id="open_html" style="width:100%; padding:15px; background:#4CAF50; color:white; border:none; border-radius:8px; font-weight:bold; font-size:16px; cursor:pointer; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
            🌐 消えない別画面で開く
          </button>
          <script>
          document.getElementById('open_html').addEventListener('click', function() {{
              const htmlContent = decodeURIComponent("{html_encoded}");
              const blob = new Blob([htmlContent], {{ type: 'text/html;charset=utf-8' }});
              const url = URL.createObjectURL(blob);
              window.open(url, '_blank');
          }});
          </script>
          """,
          height=65,
      )
    with col_jump2:
      st.download_button(
          label='📥 HTMLファイルとして保存',
          data=st.session_state['standalone_html'],
          file_name='route_result.html',
          mime='text/html',
          use_container_width=True,
      )

  # 重複除外された地点の一覧アコーディオン
  if res.get('duplicates'):
    with st.expander(
        f"⚠️ 重複として除外された地点 ({len(res['duplicates'])} 件)",
        expanded=False,
    ):
      st.caption('同じ座標（または出発地と同一座標）のため除外されたリストです。')
      dup_df = pd.DataFrame([{
          '名称/ID': d['name'],
          '緯度': d['lat'],
          '経度': d['lon'],
      } for d in res['duplicates']])
      st.dataframe(dup_df, use_container_width=True)

  # B. 並び替え済みリストのダウンロード
  st.markdown('---')
  st.subheader('📄 並び替え済みリストのダウンロード')
  st.caption(
      '元データの全情報に『訪問順』を付与して最適順に並び替えたファイルです。'
  )

  df_export = create_export_dataframe(res)

  with st.expander('👀 並び替え後のデータプレビューを表示', expanded=False):
    st.dataframe(df_export.head(10), use_container_width=True)

  csv_data = df_export.to_csv(index=False, encoding='utf-8-sig').encode(
      'utf-8-sig'
  )

  excel_buffer = io.BytesIO()
  with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
    df_export.to_excel(writer, index=False, sheet_name='最適ルート一覧')
  excel_data = excel_buffer.getvalue()

  col_dl1, col_dl2 = st.columns(2)
  with col_dl1:
    st.download_button(
        label='📥 並び替え済み CSV をDL',
        data=csv_data,
        file_name='optimized_route_list.csv',
        mime='text/csv',
        use_container_width=True,
    )
  with col_dl2:
    st.download_button(
        label='📊 並び替え済み Excel をDL',
        data=excel_data,
        file_name='optimized_route_list.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        use_container_width=True,
    )

  # C. Google Maps ナビリンク
  st.markdown('---')
  st.subheader(f'🔗 Google Maps ナビURL {mode_text}')
  st.caption(
      '💡現場から少し離れた場所から始める時は『📍'
      ' 現在地から開始』を押すと一発でナビが始まります！'
  )

  for item in res['urls_info']:
    st.markdown(
        f"**Part {item['part']}**: `{item['start']}` ➔ `{item['end']}`"
        f" ({item['count']}区間)"
    )
    col_u1, col_u2 = st.columns(2)
    with col_u1:
      st.link_button(
          f"🚗 Part {item['part']} (通常)",
          item['url'],
          use_container_width=True,
      )
    with col_u2:
      st.link_button(
          '📍 現在地から開始',
          item['current_loc_url'],
          use_container_width=True,
      )
    st.write('')

  # D. 最適化ルートマップ
  st.markdown('---')
  st.subheader('🗺️ 最適化ルートマップ')

  if 'result_map' in st.session_state:
    st_folium(
        st.session_state['result_map'],
        key='route_result_map',
        returned_objects=[],
        use_container_width=True,
        height=480,
    )
