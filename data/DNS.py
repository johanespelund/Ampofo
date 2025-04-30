import urllib.request
import tarfile
import pathlib
import progressbar
import pandas as pd

class DNS:
    def __init__(self, dataset_name):
        self.url = ( "https://filesender.renater.fr/download.php?token"
            "=fe623c25-161e-4b08-aa7d-0cde13fe69bc&files_ids=52745634" )
        self.repo_root = pathlib.Path(__file__).parent.parent
        self.folder_path = self.repo_root / "data" / "SebilleausCavity"
        self.archive_path = self.repo_root / "data" / "SebilleausCavity.tgz"
        self.data = None
        self.plot_params = {
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "none",
            "markeredgecolor": "black",
            "linestyle": "none",
            "label": "DNS",
            "zorder": 0,
        }

        self.ensure_dataset()
        self.set_dataset_path(dataset_name)

    def ensure_dataset(self):
        """
        Ensures that the dataset folder exists.
        If not, downloads and extracts it.
        """
        if self.folder_path.exists():
            print(f"Dataset already available at '{self.folder_path}'.")
            return

        self.folder_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading dataset from {self.url}.")
        print(f"This might take a while, depending on your internet connection.")
        urllib.request.urlretrieve(self.url, self.archive_path, reporthook=MyProgressBar())
        print(f"Downloaded archive to '{self.archive_path}'.\n")

        print(f"Extracting archive...")
        with tarfile.open(self.archive_path, 'r:gz') as tar:
            tar.extractall(path=self.folder_path.parent)
        print(f"Extracted dataset to '{self.folder_path.parent}'.\n")

        # Optionally, clean up archive after extraction
        self.archive_path.unlink()
        print(f"Deleted archive '{self.archive_path}'.\n")


    def set_dataset_path(self, dataset: str):
        """
        Loads the data from the dataset folder.
        dataset: str = "db_{1e10_lin,1e11_lin,1e8_lin,1p58e9_amp,1p58e9_lin}"
        dataset:
            db_1e8_lin
            db_1e10_lin
            db_1e11_lin
            db_1p58e9_amp
            db_1p58e9_lin
        """

        _dataset_path = self.folder_path / dataset
        if not _dataset_path.exists():
            raise FileNotFoundError(f"Dataset '{dataset}' not found in '{self.folder_path}'.")
        self.dataset_path = _dataset_path

    def load_data(self, filename):
        """
        Returns a pd.DataFrame containing the data from the file.
        """
        path = self.dataset_path / filename

            # First, parse manually to find the column names
        columns = None
        with open(path, 'r') as f:
            for line in f:
                line = line.lstrip()  # Remove leading whitespace
                if line.startswith('*') or not line.strip():
                    continue  # Skip comments and empty lines
                if line.startswith('|'):
                    line = line.replace('|', '')  # Remove all | characters
                    columns = [col.strip() for col in line.split()]
                    break


        if columns is None:
            raise ValueError(f"No header line with '|' found in {filename}")


        df = pd.read_csv(
                path,
                comment='*',
                sep='\s+',
                names=columns,
                skiprows=29
            )

        return df

        




class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()
