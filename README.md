# NetXplore

**NetXplore** is a web-based research platform for analyzing, visualizing, and comparing discourse networks across platforms like **WhatsApp** and **Wikipedia**.

This tool transforms linear conversations into interactive network graphs, enabling deep analysis of social structures, communication patterns, and key metrics.

---

## Features

* Upload chat or discussion exports (WhatsApp `.txt` or Wikipedia `.json`)
* Build and visualize communication networks
* Perform community detection and network analysis
* Calculate centrality metrics:
  * Degree Centrality
  * Betweenness Centrality
  * Closeness Centrality
* Apply advanced filters:
  * By date range
  * By keywords
  * By selected users
  * By minimum message length or activity
* Compare discourse across different groups
* Save and manage research history
* Export results as Excel/CSV
* Support for dual platforms: **WhatsApp** and **Wikipedia**
* Anonymize participant names for privacy

---

## Tech Stack

* **Frontend:** React, Bootstrap (hosted on AWS S3)
* **Backend:** FastAPI (Python)
* **Database:** PostgreSQL (via pgAdmin)
* **Graph Analysis:** NetworkX, custom SNA algorithms
* **Deployment:** AWS EC2 (Backend), S3 (Frontend), Nginx (reverse proxy)


