from mozarrt.cli import app


def test_cli_commands():
    """Test that the cyclopts CLI sub-apps are registered correctly."""
    # Check that the main app has the expected sub-apps
    sub_apps = [a.name[0] for a in app.subapps]
    print("Registered sub-apps:", sub_apps)
    assert "plate" in sub_apps
    assert "folder" in sub_apps
