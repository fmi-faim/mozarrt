from cyclopts import App

from mozarrt.folder import project as folder_project
from mozarrt.plate import project as plate_project

app = App()
plate_app = App(
    name="plate",
    help="Actions that take an HCS plate OME-Zarr as input",
)
folder_app = App(
    name="folder",
    help="Actions that take a folder with OME-Zarr datasets as input",
)

folder_app.command(folder_project)
plate_app.command(plate_project)
app.command(plate_app)
app.command(folder_app)
