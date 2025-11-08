import { useState } from "react";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    location: "",
    experience: "",
    degree: "",
    role: "",
    skills: [],
    weights: {
      location: 50,
      experience: 50,
      degree: 50,
      role: 50,
      skills: 50,
    },
  });

  const [users, setUsers] = useState([]); // right panel data

  const locations = ["Mumbai", "Delhi", "Bangalore", "Nagpur", "Hyderabad"];
  const degrees = ["B.Tech", "M.Tech", "PhD"];
  const roles = ["Educational", "Designer", "ML Engineer", "Developer", "Data Analyst"];
  const skillOptions = ["Python", "React", "Node.js", "C++", "Machine Learning", "SQL", "JavaScript"];

  // Input handlers
  const handleChange = (field, value) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
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

  const handleSubmit = async (e) => {
    e.preventDefault();

    const payload = { ...formData };
    console.log("📦 Sending to backend:", payload);

    // 🔥 For now, simulate fetched data
    const mockResponse = [
      {
        id: 1,
        name: "Alice Johnson",
        location: "Bangalore",
        role: "ML Engineer",
        skills: ["Python", "TensorFlow", "SQL"],
        score: 92,
      },
      {
        id: 2,
        name: "Rahul Mehta",
        location: "Delhi",
        role: "Developer",
        skills: ["React", "Node.js", "MongoDB"],
        score: 85,
      },
      {
        id: 3,
        name: "Sneha Verma",
        location: "Mumbai",
        role: "Data Analyst",
        skills: ["Python", "Excel", "PowerBI"],
        score: 79,
      },
    ];

    // pretend we fetched from backend
    setTimeout(() => {
      setUsers(mockResponse);
    }, 1000);

    // Later: uncomment this for real backend
    /*
    try {
      const res = await fetch("http://localhost:5000/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      setUsers(data.users);
    } catch (error) {
      console.error("Error fetching:", error);
    }
    */
  };

  return (
    <div className="main-container">
      {/* LEFT PANEL - Form */}
      <div className="left-panel">
        <h1>Profile Preference Form</h1>

        <form className="form-box" onSubmit={handleSubmit}>
          {/* Location */}
          <div className="form-group">
            <label>Location:</label>
            <select
              value={formData.location}
              onChange={(e) => handleChange("location", e.target.value)}
              required
            >
              <option value="">--Select Location--</option>
              {locations.map((loc) => (
                <option key={loc} value={loc}>
                  {loc}
                </option>
              ))}
            </select>
            <div className="slider-row">
              <label>Weight: {formData.weights.location}%</label>
              <input
                type="range"
                min="0"
                max="100"
                value={formData.weights.location}
                onChange={(e) => handleWeightChange("location", e.target.value)}
              />
            </div>
          </div>

          {/* Experience */}
          <div className="form-group">
            <label>Experience (Years):</label>
            <input
              type="number"
              min="0"
              placeholder="Enter years"
              value={formData.experience}
              onChange={(e) => handleChange("experience", e.target.value)}
              required
            />
            <div className="slider-row">
              <label>Weight: {formData.weights.experience}%</label>
              <input
                type="range"
                min="0"
                max="100"
                value={formData.weights.experience}
                onChange={(e) => handleWeightChange("experience", e.target.value)}
              />
            </div>
          </div>

          {/* Skills */}
          <div className="form-group">
            <label>Skills:</label>
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
            <div className="slider-row">
              <label>Weight: {formData.weights.skills}%</label>
              <input
                type="range"
                min="0"
                max="100"
                value={formData.weights.skills}
                onChange={(e) => handleWeightChange("skills", e.target.value)}
              />
            </div>
          </div>

          {/* Degree */}
          <div className="form-group">
            <label>Degree:</label>
            <select
              value={formData.degree}
              onChange={(e) => handleChange("degree", e.target.value)}
              required
            >
              <option value="">--Select Degree--</option>
              {degrees.map((deg) => (
                <option key={deg} value={deg}>
                  {deg}
                </option>
              ))}
            </select>
            <div className="slider-row">
              <label>Weight: {formData.weights.degree}%</label>
              <input
                type="range"
                min="0"
                max="100"
                value={formData.weights.degree}
                onChange={(e) => handleWeightChange("degree", e.target.value)}
              />
            </div>
          </div>

          {/* Role */}
          <div className="form-group">
            <label>Role:</label>
            <select
              value={formData.role}
              onChange={(e) => handleChange("role", e.target.value)}
              required
            >
              <option value="">--Select Role--</option>
              {roles.map((role) => (
                <option key={role} value={role}>
                  {role}
                </option>
              ))}
            </select>
            <div className="slider-row">
              <label>Weight: {formData.weights.role}%</label>
              <input
                type="range"
                min="0"
                max="100"
                value={formData.weights.role}
                onChange={(e) => handleWeightChange("role", e.target.value)}
              />
            </div>
          </div>

          <button type="submit" className="submit-btn">
            Search Profiles
          </button>
        </form>
      </div>

      {/* RIGHT PANEL - User Cards */}
      <div className="right-panel">
        <h2>Matched Profiles</h2>

        {users.length === 0 ? (
          <p className="no-data">No profiles yet. Submit form to fetch.</p>
        ) : (
          users.map((user) => (
            <div key={user.id} className="user-card">
              <h3>{user.name}</h3>
              <p><b>Role:</b> {user.role}</p>
              <p><b>Location:</b> {user.location}</p>
              <p><b>Score:</b> {user.score}%</p>
              <div className="skill-tags">
                {user.skills.map((skill, i) => (
                  <span key={i} className="tag">{skill}</span>
                ))}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default App;
