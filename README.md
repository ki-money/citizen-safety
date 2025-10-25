# 🚨 Citizen Safety Reporting Platform

A comprehensive web-based incident reporting system that enables citizens to anonymously report crimes and safety incidents to local police stations in Nakuru County, Kenya. The platform features AI-powered spam detection, automatic language detection (English/Kiswahili), geocoding, and real-time analytics.

## ✨ Features

### For Citizens
- 🔒 **Anonymous Reporting** - Submit incidents without revealing personal information
- 🌍 **Multilingual Support** - Auto-detection of English and Kiswahili
- 📍 **Smart Geocoding** - Automatic location detection with fuzzy matching
- 📱 **Mobile Responsive** - Works seamlessly on any device
- 🎯 **Report Tracking** - Track your report status using a unique ID
- 📸 **Media Upload** - Attach photos or videos (up to 10MB)

### For Police Officers
- 📊 **Real-time Dashboard** - View all reports for your constituency
- 🔍 **AI Analytics** - Anomaly detection for urgent incidents
- 🌐 **Auto-Translation** - Translate reports between English and Kiswahili
- 📥 **PDF Export** - Download individual reports as PDF
- 🗺️ **Hotspot Mapping** - Interactive map with crime density analysis
- 🎯 **Smart Recommendations** - AI-powered patrol suggestions
- ⚡ **Quick Response** - Respond to reports with status updates

### For Administrators
- 👥 **Station Management** - Add, edit, activate/deactivate police stations
- 📈 **System Analytics** - Comprehensive statistics and metrics
- 🛡️ **Spam Detection** - AI-powered spam filtering (configurable thresholds)
- 📊 **Bulk Export** - Download all reports as PDF or CSV
- 📝 **Audit Logs** - Complete system activity tracking
- ⚙️ **System Settings** - Configure categories, thresholds, and more

### AI & Machine Learning Features
- 🤖 **Spam Detection** - Multi-factor analysis (60+ confidence scoring)
- 🔍 **Anomaly Detection** - Identifies critical/urgent incidents
- 📍 **Smart Geocoding** - 3-tier location resolution system
- 🗺️ **Hotspot Clustering** - Hierarchical clustering with Ward's method
- 📊 **Density Analysis** - Risk level classification (Critical/High/Medium/Low)
- 📈 **Trend Analysis** - Time-series pattern detection
- 🎯 **Patrol Recommendations** - Data-driven deployment suggestions

## 🏗️ System Architecture

┌─────────────────────────────────────────────────────────────┐
│                     Citizen Interface                        │
│  (Report Submission, Tracking, Map View, Language Toggle)   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     Flask Application                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Routes     │  │  Middleware  │  │   Security   │      │
│  │  (app.py)    │  │   (Auth)     │  │   (Bleach)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────┬────────────────────────────────────────┬─────┘
               │                                        │
               ▼                                        ▼
┌─────────────────────────────┐    ┌─────────────────────────────┐
│     AI Analytics Engine     │    │      MongoDB Database       │
│  ┌────────────────────────┐ │    │  ┌────────────────────────┐ │
│  │  Spam Detection        │ │    │  │  reports                │ │
│  │  Anomaly Detection     │ │    │  │  police_stations        │ │
│  │  Geocoding             │ │    │  │  responses              │ │
│  │  Clustering            │ │    │  │  hotspots               │ │
│  │  Translation           │ │    │  │  audit_logs             │ │
│  │  Trend Analysis        │ │    │  │  system_settings        │ │
│  └────────────────────────┘ │    │  │  admin_users            │ │
└─────────────────────────────┘    │  └────────────────────────┘ │
                                   └─────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│              Police & Admin Dashboards                       │
│  (Analytics, Reports, Maps, Station Management, Exports)    │
└─────────────────────────────────────────────────────────────┘

git clone https://github.com/yourusername/nakuru-safety-platform.git



**Option B: MongoDB Atlas (Cloud)**
1. Create free account at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create a cluster
3. Get connection string
4. Update `MONGO_URI` in `database.py`



## 📸 Screenshots

### Citizen Interface
- **Homepage**: Report submission, tracking, and safety information
- **Report Form**: Easy-to-use multilingual reporting interface
- **Track Report**: Real-time status updates

### Police Dashboard
- **Analytics**: Real-time statistics and trends
- **Reports Table**: Comprehensive report management
- **Interactive Map**: Crime hotspot visualization with clustering

### Admin Dashboard
- **Station Management**: Add/edit/activate police stations
- **System Overview**: Platform-wide statistics
- **Bulk Export**: Download all reports as PDF or CSV

## 🛠️ Technologies Used

### Backend
- **Flask 2.3.3** - Web framework
- **MongoDB 4.6.3** - NoSQL database
- **PyMongo** - MongoDB driver
- **NumPy 2.0.2** - Numerical computing
- **SciPy 1.13.1** - Scientific computing (clustering)

### Frontend
- **Bootstrap 5.3.3** - UI framework
- **Leaflet.js** - Interactive maps
- **Leaflet.markercluster** - Map clustering

### AI & Analytics
- **googletrans** - Language detection & translation
- **deep-translator** - Fallback translation
- **ReportLab** - PDF generation
- **Custom AI Engine** - Spam detection, anomaly detection, geocoding


## 🔐 Security Features

### Input Validation
- ✅ Sanitization with Bleach
- ✅ Maximum length limits
- ✅ File type validation
- ✅ File size limits (10MB)
- ✅ MIME type checking

### Authentication
- ✅ Bcrypt password hashing
- ✅ Secure session management
- ✅ Role-based access control
- ✅ Session timeout (1 hour)
- ✅ CSRF protection

### Data Protection
- ✅ Anonymous reporting (no PII collected)
- ✅ Secure file uploads
- ✅ Database connection encryption
- ✅ Audit logging for all actions

### Spam Prevention
- ✅ AI-powered spam detection
- ✅ Configurable thresholds
- ✅ Auto-rejection of high-risk reports
- ✅ GPS validation
- ✅ Location verification

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request


## 📞 Support

### Contact Information
- **Emergency**: 0725646760
- **Email**: www.magwekim@gmail.com
- **Website**: https://nakuru-safety.ke


### Reporting Issues
Found a bug? [Open an issue](https://github.com/yourusername/nakuru-safety-platform/issues)

**Made with ❤️ for Nakuru County, Kenya**

*Keep Nakuru Safe - One Report at a Time* 🇰🇪