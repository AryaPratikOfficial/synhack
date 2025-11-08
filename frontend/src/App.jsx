import { useState } from "react";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    location: "",
    experience: { min: "", max: "" },
    degree: "",
    role: "",
    ctc_range: { min: "5", max: "10" },
    skills: [],
    weights: {
      location: 50,
      experience: 50,
      degree: 50,
      role: 50,
      skills: 50,
      ctc_range: 50,
    },
    page: 1,
    per_page: 20,
  });

  const [users, setUsers] = useState([]);
  const [pagination, setPagination] = useState(null);
  const [loading, setLoading] = useState(false);

  const locations = ["Mumbai", "Delhi", "Bangalore", "Nagpur", "Hyderabad"];
  const degrees = ["B.Tech", "M.Tech", "PhD"];
  const roles = ["Educational", "Designer", "ML Engineer", "Developer", "Data Analyst"];
  const skillOptions = ["Python", "React", "Node.js", "C++", "Machine Learning", "SQL", "JavaScript"];

  // ===== HANDLERS =====
  const handleChange = (field, value) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleExperienceChange = (type, value) => {
    setFormData((prev) => ({
      ...prev,
      experience: { ...prev.experience, [type]: value },
    }));
  };

  const handleCTCChange = (type, value) => {
    setFormData((prev) => ({
      ...prev,
      ctc_range: { ...prev.ctc_range, [type]: value },
    }));
  };

  const handleSkillChange = (skill) => {
    setFormData((prev) => {
      const alreadySelected = prev.skills.includes(skill);
      const updatedSkills = alreadySelected
        ? prev.skills.filter((s) => s !== skill)
        : [...prev.skills, skill];
      return { ...prev, skills: updatedSkills };
    });
  };

  const handleWeightChange = (field, value) => {
    setFormData((prev) => ({
      ...prev,
      weights: { ...prev.weights, [field]: parseInt(value) },
    }));
  };

  // ===== API CALL =====
  const handleSubmit = async (e) => {
    e.preventDefault();
    const payload = { ...formData };
    console.log("📦 Sending to backend:", payload);

    setLoading(true);

    try {
      const response = await fetch("http://100.71.15.108:5000/api/candidates/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error("Failed to connect to backend");

      const data = await response.json();
      console.log("✅ Response from backend:", data);

      // ✅ Correct mapping for backend structure
      if (data.success && Array.isArray(data.data)) {
        setUsers(data.data);
        setPagination(data.pagination || null);
      } else {
        setUsers([]);
      }
    } catch (error) {
      console.error("❌ Error:", error);
      alert("Backend connection failed. Check console or server.");
    } finally {
      setLoading(false);
    }
  };

  // ===== UI =====
  return (
    <div className="main-container">
      {/* ===== LEFT PANEL ===== */}
      <div className="left-panel">
        <h1 className="title">🔧 Profile Preference Setup</h1>

        <form className="form-box" onSubmit={handleSubmit}>
          {/* Location */}
          <div className="form-group glass">
            <label>Location</label>
            <select
              value={formData.location}
              onChange={(e) => handleChange("location", e.target.value)}
              required
            >
              <option value="">Select location...</option>
              {locations.map((loc) => (
                <option key={loc}>{loc}</option>
              ))}
            </select>
            <input
              type="range"
              min="0"
              max="100"
              value={formData.weights.location}
              onChange={(e) => handleWeightChange("location", e.target.value)}
            />
            <small>Weight: {formData.weights.location}%</small>
          </div>

          {/* Experience */}
          <div className="form-group glass">
            <label>Experience Range (Years)</label>
            <div className="exp-range">
              <input
                type="number"
                min="0"
                placeholder="From"
                value={formData.experience.min}
                onChange={(e) => handleExperienceChange("min", e.target.value)}
                required
              />
              <span className="to-text">to</span>
              <input
                type="number"
                min="0"
                placeholder="To"
                value={formData.experience.max}
                onChange={(e) => handleExperienceChange("max", e.target.value)}
                required
              />
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={formData.weights.experience}
              onChange={(e) => handleWeightChange("experience", e.target.value)}
            />
            <small>Weight: {formData.weights.experience}%</small>
          </div>

          {/* CTC Range */}
          <div className="form-group glass">
            <label>CTC Range (LPA)</label>
            <div className="exp-range">
              <input
                type="number"
                min="0"
                placeholder="Min"
                value={formData.ctc_range.min}
                onChange={(e) => handleCTCChange("min", e.target.value)}
              />
              <span className="to-text">to</span>
              <input
                type="number"
                min="0"
                placeholder="Max"
                value={formData.ctc_range.max}
                onChange={(e) => handleCTCChange("max", e.target.value)}
              />
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={formData.weights.ctc_range}
              onChange={(e) => handleWeightChange("ctc_range", e.target.value)}
            />
            <small>Weight: {formData.weights.ctc_range}%</small>
          </div>

          {/* Skills */}
          <div className="form-group glass">
            <label>Skills</label>
            <div className="skills-list">
              {skillOptions.map((skill) => (
                <label key={skill} className="skill-item">
                  <input
                    type="checkbox"
                    checked={formData.skills.includes(skill)}
                    onChange={() => handleSkillChange(skill)}
                  />
                  {skill}
                </label>
              ))}
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={formData.weights.skills}
              onChange={(e) => handleWeightChange("skills", e.target.value)}
            />
            <small>Weight: {formData.weights.skills}%</small>
          </div>

          {/* Degree */}
          <div className="form-group glass">
            <label>Degree</label>
            <select
              value={formData.degree}
              onChange={(e) => handleChange("degree", e.target.value)}
              required
            >
              <option value="">Select degree...</option>
              {degrees.map((deg) => (
                <option key={deg}>{deg}</option>
              ))}
            </select>
            <input
              type="range"
              min="0"
              max="100"
              value={formData.weights.degree}
              onChange={(e) => handleWeightChange("degree", e.target.value)}
            />
            <small>Weight: {formData.weights.degree}%</small>
          </div>

          {/* Role */}
          <div className="form-group glass">
            <label>Role</label>
            <select
              value={formData.role}
              onChange={(e) => handleChange("role", e.target.value)}
              required
            >
              <option value="">Select role...</option>
              {roles.map((r) => (
                <option key={r}>{r}</option>
              ))}
            </select>
            <input
              type="range"
              min="0"
              max="100"
              value={formData.weights.role}
              onChange={(e) => handleWeightChange("role", e.target.value)}
            />
            <small>Weight: {formData.weights.role}%</small>
          </div>

          <button className="submit-btn neon" type="submit">
            {loading ? "⏳ Finding..." : "🚀 Find Matches"}
          </button>
        </form>
      </div>

      {/* ===== RIGHT PANEL ===== */}
      <div className="right-panel">
        <h2 className="subtitle">📋 Matched Profiles</h2>

        {loading ? (
          <div className="placeholder"><p>Loading matches...</p></div>
        ) : users.length === 0 ? (
          <div className="placeholder"><p>No profiles yet. Submit to view matches.</p></div>
        ) : (
          <div>
            <div className="card-container">
              {users.map((profile, index) => (
                <div key={index} className="user-card glass-card">
                  <h3>Candidate #{index + 1}</h3>
                  <p><b>City:</b> {profile.city}</p>
                  <p><b>Experience:</b> {profile.experience_years} yrs</p>
                  <p><b>CTC Expectation:</b> ₹{profile.ctc_expectation_k}K</p>
                  <p className="score">
                    <b>Final Score:</b> {profile.f_composite_score?.toFixed(2)}%
                  </p>
                  <small>Composite Score: {profile.composite_score?.toFixed(3)}</small>
                </div>
              ))}
            </div>

            {/* Pagination Info */}
            {pagination && (
              <div className="pagination-info">
                <p>
                  Page {pagination.page} of {pagination.total_pages} —{" "}
                  Total {pagination.total_items} profiles
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
