from cyclopts import App

from mozarrt.collect import create_plate_project, folder, plate
from mozarrt.folder import project

app = App()
plate_app = App(
    name="plate",
    help="Actions that take an HCS plate OME-Zarr as input",
)
folder_app = App(
    name="folder",
    help="Actions that take a folder with OME-Zarr datasets as input",
)

folder_app.command(project)
app.command(plate_app)
app.command(folder_app)
