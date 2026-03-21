---
name: Use Django admin commands for boilerplate
description: User prefers using django-admin startproject and startapp to generate Django boilerplate, not writing manage.py and settings.py by hand
type: feedback
---

Use `django-admin startproject` and `python manage.py startapp` to generate Django boilerplate files. Only edit/extend those generated files rather than writing Django scaffolding from scratch.

**Why:** User explicitly corrected this approach — Django's own tooling should produce the initial manage.py, settings.py, urls.py, wsgi.py, asgi.py, etc.

**How to apply:** Any time a new Django project or app needs to be initialized, run the appropriate management command first, then edit the generated output.
