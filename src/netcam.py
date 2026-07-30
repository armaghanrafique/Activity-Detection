"""Network (IP / Ethernet) camera support.

From just an IP address + login, this can:
  * build candidate RTSP / HTTP stream URLs (common paths for major brands),
  * ask an ONVIF camera for its exact RTSP URI,
  * probe candidates and open the first one that returns a real frame,
  * discover ONVIF cameras on the LAN (WS-Discovery).

Browsers cannot open RTSP, so the server opens the stream with OpenCV (ffmpeg)
and the pipeline re-streams the annotated result to the browser.
"""

import os
import socket
import threading
import time
from urllib.parse import quote, urlparse, urlunparse

import cv2
import numpy as np


def reachable(host, port, timeout=1.5):
    """Fast TCP check so an unreachable IP/port fails in ~1s instead of hanging."""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False


def _safe_int(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _url_host_port(url, ip):
    """Host + port from a URL, tolerating a non-numeric/garbage port."""
    try:
        p = urlparse(url)
        host = p.hostname or ip
        try:
            port = p.port
        except ValueError:
            port = None
        if not port:
            port = 554 if p.scheme == "rtsp" else 80
        return host, port
    except Exception:
        return ip, 554

# Make ffmpeg's RTSP robust: force TCP transport and a socket timeout so a bad
# URL/IP fails fast instead of hanging forever. (microseconds)
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000"
)

# Common RTSP stream paths across major brands + generic fallbacks.
RTSP_PATHS = [
    "/Streaming/Channels/101",                 # Hikvision main
    "/Streaming/Channels/102",                 # Hikvision sub
    "/cam/realmonitor?channel=1&subtype=0",    # Dahua main
    "/cam/realmonitor?channel=1&subtype=1",    # Dahua sub
    "/h264Preview_01_main",                     # Reolink
    "/axis-media/media.amp",                    # Axis
    "/stream1", "/stream2",                     # TP-Link / generic
    "/live", "/live/main", "/live/ch00_0",
    "/videoMain", "/media/video1", "/ch01.264", "/11", "/1", "/",
]
HTTP_PATHS = [
    "/video.mjpg", "/mjpg/video.mjpg", "/video.cgi",
    "/videostream.cgi", "/axis-cgi/mjpg/video.cgi", "/video",
]


def _auth(user, pw):
    return f"{quote(user)}:{quote(pw)}@" if user else ""


def _inject_auth(url, user, pw):
    """Insert user:pass@ into a URL that has no credentials."""
    if not user:
        return url
    p = urlparse(url)
    if "@" in (p.netloc or ""):
        return url
    netloc = f"{quote(user)}:{quote(pw)}@{p.netloc}"
    return urlunparse((p.scheme, netloc, p.path, p.params, p.query, p.fragment))


def build_candidates(cfg):
    """Build the list of stream URLs to try from a config dict.

    cfg keys: url (full, optional), ip, port (optional), user, password,
    protocol ('rtsp' | 'http' | 'auto'), path (optional custom).
    """
    full = (cfg.get("url") or "").strip()
    if full:
        return [full]

    ip = (cfg.get("ip") or "").strip()
    user = cfg.get("user", "") or ""
    pw = cfg.get("password", "") or ""
    proto = (cfg.get("protocol") or "auto").lower()
    path = (cfg.get("path") or "").strip()
    auth = _auth(user, pw)
    out = []

    def add(scheme, default_port, paths):
        port = cfg.get("port") or default_port
        base = f"{scheme}://{auth}{ip}:{port}"
        chosen = [path if path.startswith("/") else "/" + path] if path else paths
        out.extend(base + p for p in chosen)

    if proto in ("rtsp", "auto"):
        add("rtsp", 554, RTSP_PATHS)
    if proto in ("http", "auto"):
        add("http", 80, HTTP_PATHS)
    return out


def open_stream(url, warmup=2.5, budget=7.0):
    """Open a stream URL with a hard timeout. Returns (cap, first_frame) or
    (None, None). A blocking/half-open connection is abandoned after ``budget``."""
    out = {"cap": None, "frame": None}

    def _work():
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap or not cap.isOpened():
                if cap:
                    cap.release()
                return
            t0 = time.time()
            while time.time() - t0 < warmup:
                ok, f = cap.read()
                if ok and f is not None and float(np.mean(f)) > 3.0:
                    out["cap"], out["frame"] = cap, f
                    return
            cap.release()
        except Exception:
            pass

    th = threading.Thread(target=_work, daemon=True)
    th.start()
    th.join(budget)
    if th.is_alive():
        return None, None
    return out["cap"], out["frame"]


def onvif_stream_uri(ip, onvif_port, user, pw):
    """Ask an ONVIF camera for its exact RTSP stream URI (best case)."""
    try:
        from onvif import ONVIFCamera

        cam = ONVIFCamera(ip, int(onvif_port or 80), user, pw)
        media = cam.create_media_service()
        profiles = media.GetProfiles()
        if not profiles:
            return None
        req = media.create_type("GetStreamUri")
        req.ProfileToken = profiles[0].token
        req.StreamSetup = {"Stream": "RTP-Unicast",
                           "Transport": {"Protocol": "RTSP"}}
        uri = media.GetStreamUri(req).Uri
        return _inject_auth(uri, user, pw)
    except Exception as exc:
        print(f"[netcam] ONVIF stream-uri failed for {ip}: {exc}")
        return None


def resolve(cfg):
    """Find a working stream. Returns (url, cap, frame) or (None, None, None).

    Order: explicit URL -> ONVIF GetStreamUri -> common-path probing.
    """
    # 1) explicit full URL
    full = (cfg.get("url") or "").strip()
    if full:
        cap, frame = open_stream(full)
        if cap is not None:
            return full, cap, frame

    ip = (cfg.get("ip") or "").strip()
    if not ip:
        return None, None, None
    user, pw = cfg.get("user", ""), cfg.get("password", "")

    # 2) ONVIF exact URI (uses the ONVIF service port, usually 80) — only if reachable
    onvif_port = _safe_int(cfg.get("onvif_port"), 80)
    if reachable(ip, onvif_port):
        uri = onvif_stream_uri(ip, onvif_port, user, pw)
        if uri:
            cap, frame = open_stream(uri)
            if cap is not None:
                return uri, cap, frame

    # 3) probe common paths, skipping any host:port that isn't reachable
    checked = {}
    for url in build_candidates(cfg):
        host, port = _url_host_port(url, ip)
        key = (host, port)
        if key not in checked:
            checked[key] = reachable(host, port)
        if not checked[key]:
            continue
        cap, frame = open_stream(url)
        if cap is not None:
            return url, cap, frame
    return None, None, None


def discover(timeout=4):
    """WS-Discovery scan for ONVIF cameras on the LAN. Returns [{ip, name}]."""
    found, seen = [], set()
    try:
        from wsdiscovery import QName
        from wsdiscovery.discovery import ThreadedWSDiscovery

        wsd = ThreadedWSDiscovery()
        wsd.start()
        nvt = QName("http://www.onvif.org/ver10/network/wsdl",
                    "NetworkVideoTransmitter")
        services = wsd.searchServices(types=[nvt], timeout=timeout)
        for s in services:
            for xaddr in s.getXAddrs():
                host = urlparse(xaddr).hostname
                if host and host not in seen:
                    seen.add(host)
                    found.append({"ip": host, "name": "ONVIF Camera"})
        wsd.stop()
    except Exception as exc:
        print(f"[netcam] discovery error: {exc}")
    return found
