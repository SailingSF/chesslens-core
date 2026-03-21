# Generated manually for American spelling consistency

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("game", "0001_initial"),
    ]

    operations = [
        migrations.RenameField(
            model_name="savedgame",
            old_name="analysed_at",
            new_name="analyzed_at",
        ),
    ]
