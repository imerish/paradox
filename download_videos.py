#Downloading videos
import yt_dlp

urls = [
    "https://www.youtube.com/shorts/_i3-UNFb_ao",
    "https://www.youtube.com/shorts/k98X_rBfFZM",
]

options = {
    "format": "best[video_ext=mp4][audio_ext=m4a]/best",
    "outtmpl": "videos/%(title)s.%(ext)s",
}

with yt_dlp.YoutubeDL(options) as ydl:
    ydl.download(urls)