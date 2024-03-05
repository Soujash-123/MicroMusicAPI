import csv
import youtube_dl

def download_audio(track_name):
    # Specify options for youtube_dl
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{track_name}.mp3',
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([f"ytsearch:{track_name} audio"])
        except Exception as e:
            print(f"Error downloading {track_name}: {e}")

def download_songs_from_csv(csv_file, column_name):
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            track_name = row[column_name]
            download_audio(track_name)


if __name__ == "__main__":
    csv_file = "top50.csv"  # Provide the path to your CSV file
    column_name = "Track.Name"  # Specify the column name containing track names
    download_songs_from_csv(csv_file, column_name)
