"""
app.py
------
Streamlit frontend for the IT Ticket Classifier.
Sends ticket details to the FastAPI /predict endpoint and displays the result.

"""

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL = "http://localhost:8000/predict"

# ---------------------------------------------------------------------------
# Dropdown options
# Replace BUSINESS_SERVICES and SERVICE_OFFERINGS with your actual values
# ---------------------------------------------------------------------------

BUSINESS_SERVICES = ['Applications', 'Networks', 'Hosting and Storage', 'End User Services', 'EcoWorld (EWL)', 'Security and Compliance', 'Service Operations', 'Service Desk', 'Data & Analytics', 'Fortem', 'Enterprise Architecture', 'Digital Communications', 'Other']
   

SERVICE_OFFERINGS = ['Azure Data Factory', 'Network', 'BT', 'Server', 'CODA', 'Azure', 'Software', 'Printer', 'Email (EWL)', 'Mailbox', 'Subcontract Procurement', 'Laptop', 'Email/Outlook', 'Mouse', 'Power Project Enterprise', 'Site Set Up', 'ServiceNow', 'Request', 'Meeting Room', 'Dynamics CE', 'Microsoft Teams', 'Hardware', 'Business Objects', 'Encryption Certificate Install', 'Asta Powerprojects', 'Security Alerts', 'Project Purchase Order System', 'Readsoft', 'Leaver', 'Monitor', 'Word', 'Password Reset', 'Files and Folders', 'Mobile Phone (EWL)', 'Print Driver', 'Sharepoint', 'Power BI', 'Site Connections', 'Adobe', 'Permissions', 'Site Setup', 'Willmott Dixon University', 'ResourceLink/MyView', 'Add/Remove User', 'Supply Chain', 'Performance Measurement', 'MFA', 'Spam', 'Holiday Planner', 'OneDrive', 'Teams Telephony', 'Active Directory', 'Image Laptop', 'Laptop (EWL)', 'Project Portal', 'Nialli Visual Planner', 'The Hub', 'SQL Server', 'Firewall', 'Microsoft Office 365 (EWL)', 'Veritas', 'Mobile Outlook', 'Goods Partner Invoicing', 'Power BI Licence', 'Mi Aftercare', 'Scanner', 'Network Access', 'BI Datawarehouse', 'Bluebeam', 'Cisco AnyConnect VPN', 'Common Data', 'MyLearning', 'Encryption Certificate Recovery', 'Keyboard', 'Resource Planner', 'O365 - Outlook', 'Charger', 'DocuSign Direct Portal', 'CCTV - Comms Room Surveillance', 'CostX', 'Commercial Portal', 'Zscaler', 'Viewpoint For Projects', 'Bit Locker Reset', 'SCCM', 'Mobile Phone', 'Desktop', 'Comms Room', 'Whitelist', 'Tablet', 'Expenses', 'Mobile Comms', 'Email Archive', 'Router', 'AutoDesk', 'BI Power Automate Flows', 'Product Data System (PDS)', 'Distribution List', 'Papercut PIN', 'Group Policy', 'SQL Database', 'Procurement & Vendor Management', 'Revitzo', 'O365 - Triage ', 'Excel', 'Exchange Online', 'SharePoint Migration', 'New Starter', 'Chrome', 'Conquest', 'Exchange', 'VPN', 'AutoCAD', 'Memory Card (SD)', 'MyEpayWindow (EWL)', 'Website (EWL) ', 'Headset', 'FieldView', 'CVC', 'PCF Application', 'WiFi', '4Projects', 'SharePoint (EWL) ', 'Scanning Account', 'Solibri', 'Email Service', 'Adobe Suite (EWL)', 'CRM', 'Microsoft Sway', 'O365 - Encryption', 'IT Induction', 'Subcontractor Portal', 'Microsoft InTune', 'EcoWorld', 'Domain Controller', 'Locked', 'Desk Booking App', 'Microsoft', 'Yellow Book Audit', 'Asta Powerprojects (EWL)', 'Stolen Equipment', 'Time Reporting', 'Power Apps', 'PDF Creator', 'Anti-Virus', 'File Storage (EWL)', 'INVU', 'DMS', 'Modum', 'Office (Package)', 'PowerPoint', 'CIS', 'DTS (Design Information Tracking System)', 'Docking Station', 'Builders Profile', 'Site Network Connectivity (EWL)', 'OneNote', 'MiProject', 'Camera', 'Access Key', 'Site Closure', 'Phone', 'One Map', 'SolarWinds', 'Folder Permissions', 'Google API', 'Hard Drive', 'Oculo', 'WebCam', 'Scanning Setup', 'Builders Profile Service', 'Client Invoicing', 'Employment Status Checker (IR35)', 'Mapping', 'Internet Explorer', 'Backup Tapes', 'All Safe To Work', 'Project', 'BIM Link', 'File Server', 'Charity Build', 'Microsoft Planner', 'Company Portal App', 'Cirrus Contact Centre', 'LAN', 'Capital Employed', 'Network Cable', 'HouseBuilding', 'Windows 11', 'XenSam', 'Speaker', 'Thycotic', 'Subcontractor Management Fee', 'Datacentre Servers', 'Switch', 'DNS', 'Website Blocked', 'GPI Email Service', 'Phone Lines', 'IIS', 'Projector', 'Excel Automation', 'CVC_Excel', 'Commercial Dashboard', 'Archiving', 'FTP', 'Software Licences', 'Yammer O365', 'NXG', 'Turnstile/Site Access', 'Apple', 'Java', 'IT Messenger Tool', 'SketchUp', 'Camtasia', 'Franking Machine', 'BRE Smartwaste', 'Azure DevOps', 'Renewals', 'Access Review (Audit)', 'Access Point', 'Bluebeam (EWL)', 'Axomic OpenAsset (EWL)', 'Printing & Scanning (EWL)', 'Print Server', 'Remote Access (EWL)', 'Meeting Room Technology (EWL)', 'Dynamics F&O', 'Battery', 'GPI Importer Service', 'DWG Trueview', 'Assets', 'Contract Cost', 'Microsoft Teams (EWL)', 'ODBC', 'willmottdixon.co.uk', 'Gifts and Hospitality', 'COO Reporting', 'Employee App', 'Creative Cloud', 'Total Mobile (Fortem)', 'Qtax\Qtac', 'Spam Titan', 'Device (Fortem Service Del)', 'Datawarehouse - DO NOT USE', 'Windows 10', 'ASR Backup', 'Microsoft Stream', 'Restore', 'Share', 'Television (TV)', 'Mi Risk', 'DocuSign (EWL)', 'Azure Backup', 'Power Supply', 'Network Port', 'Report Creation & Amendments (EWL)', 'Toner/Fuser/Transer Kit', '3rd Party Client', 'Line of Balance', 'External HDD', 'Subcontractor Admin Portal', 'HDMI cable', 'UPS', 'App Map', 'Cyber Security Consultancy (EWL)', 'Lumion', 'Dalux', 'BI Servers', 'Consultancy', 'IPAD (tablet)', 'Board Pad', 'ePAM', 'TSM Backup', 'DHCP', 'Malware', 'Sypro Contract Manager', 'Folder Permissions_Fortem', 'Investigation', 'AirCon', 'Security Hardening', 'Other_Fortem', 'Mi Performance', 'Primavera', 'PIX4Dmapper', 'HDD', 'Power Automate', 'Mi Audit', 'Mi SHE', 'Desk Phone', 'TMC (The Miles Consultancy)', 'Visio', 'Symantec', 'New Starter System', 'Mi Customer', 'F-Secure', 'Monday.com', 'Wireless Keyboard & Mouse', '20H2', 'Synchro', 'Mi Gatekeeper', 'ManRep - Excel', 'Mover (Change of job)', 'Enscape', 'POCAR (EWL)', 'Office Network Connectivity (EWL)', 'Life Assurance', 'Digital Communications (EWL)', 'Mi Pre-Enrolment', 'Electricity Supply Application Form', 'Device', 'Microsoft Bookings', 'Clickshare', 'Knowledge Hub', 'WeTransfer', 'MSite', 'Mailbox Full', 'theyellowbook.design', 'Digital Warranties', 'Autodesk AEC Collection (EWL)', 'Canva', 'Azure Synapse', 'Payroll Costing']


URGENCY_OPTIONS  = ["1 - High", "2 - Medium", "3 - Low"]
PRIORITY_OPTIONS = ["1 - Critical", "2 - High", "3 - Moderate", "4 - Low", "5 - Planning"]
IMPACT_OPTIONS   = ["1- High", "2 - Medium", "3 - Low"]
INCIDENT_TYPES   = ["Incident", "Request", "Problem"]

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="IT Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)

st.title("🎫 IT Ticket Classifier")
st.markdown("Fill in the ticket details below to predict which team should handle it.")
st.divider()

# ---------------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------------

with st.form("ticket_form"):

    st.subheader("Ticket Details")

    short_description = st.text_input(
        "Short Description *",
        placeholder="e.g. Laptop not connecting to VPN"
    )

    description = st.text_area(
        "Description *",
        placeholder="Describe the issue in detail...",
        height=150
    )

    st.subheader("Service Information")

    col1, col2 = st.columns(2)
    with col1:
        business_service = st.selectbox("Business Service *", BUSINESS_SERVICES)
    with col2:
        service_offering = st.selectbox("Service Offering *", SERVICE_OFFERINGS)

    st.subheader("Priority & Impact")

    col3, col4, col5 = st.columns(3)
    with col3:
        urgency = st.selectbox("Urgency *", URGENCY_OPTIONS)
    with col4:
        priority = st.selectbox("Priority *", PRIORITY_OPTIONS)
    with col5:
        impact = st.selectbox("Impact *", IMPACT_OPTIONS)

    incident_type = st.selectbox("Incident Type *", INCIDENT_TYPES)

    submitted = st.form_submit_button("🔍 Predict Assignment Group", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

if submitted:
    if not short_description or not description:
        st.error("Please fill in both Short Description and Description.")
    else:
        payload = {
            "short_description": short_description,
            "description":       description,
            "business_service":  business_service,
            "service_offering":  service_offering,
            "urgency":           urgency,
            "priority":          priority,
            "impact":            impact,
            "incident_type":     incident_type,
        }

        with st.spinner("Predicting..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()

                st.divider()
                st.success("Prediction Complete!")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(
                        label="Predicted Assignment Group",
                        value=result["predicted_assignment_group"]
                    )
                with col_b:
                    st.metric(
                        label="Confidence",
                        value=f"{result['confidence'] * 100:.1f}%"
                    )

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the API. Make sure FastAPI is running on port 8000.")
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e.response.text}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption("IT Ticket Classifier · Powered by LightGBM")
