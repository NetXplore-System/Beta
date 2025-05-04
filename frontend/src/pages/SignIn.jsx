import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useDispatch } from "react-redux";
import { setUser } from "../redux/user/userSlice";
import Button from "react-bootstrap/Button";
import Form from "react-bootstrap/Form";
import Container from "react-bootstrap/Container";
import OAuth from "../components/OAuth.jsx";
import Card from "react-bootstrap/Card";
import Image from "react-bootstrap/Image";
import logo from "../assets/Logo.png";
import { LoaderIcon } from "react-hot-toast";
import { AiOutlineLoading } from "react-icons/ai";

 
const SignIn = () => {
  const [formData, setFormData] = useState({ email: "", password: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const dispatch = useDispatch();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
  
    const { email, password } = formData;
  
    try {
      setLoading(true);
      const response = await fetch(import.meta.env.VITE_API_URL+"/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });
  
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Login failed");
      }
  
      const data = await response.json();
      dispatch(setUser(data)); 
      // navigate("/profile");
      navigate("/choose-platform");
    } catch (err) {
      setError(err.message);
    }finally {
      setLoading(false);
    }
  };
  

  return (
    <Container
    className="d-flex justify-content-center align-items-center pb-4"
    style={{ minHeight: "100vh" }}
  >
    <Card style={{ width: "100%", maxWidth: "500px",borderRadius: "20px" }} className="shadow">
      <Card.Body className="p-4">
        <div className="text-center mb-4">
          <Image 
            src={logo}
            alt="Logo" 
            width="150" 
            className="mb-3" 
          />
          <h3>Sign In</h3>
        </div>
        
        {error && <p className="text-danger text-center mb-3">{error}</p>}

        <Form onSubmit={handleSubmit}>
          <Form.Group className="mb-3" controlId="formBasicEmail">
            <Form.Label>Email address</Form.Label>
            <Form.Control
              type="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              placeholder="Enter email"
              required
            />
          </Form.Group>

          <Form.Group className="mb-3" controlId="formBasicPassword">
            <Form.Label>Password</Form.Label>
            <Form.Control
              type="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              placeholder="Password"
              required
            />
          </Form.Group>

          <Button disabled={loading} variant="primary" type="submit" className="w-100 mb-3"   style={{ backgroundColor: "#050d2d", borderColor: "#050d2d" }}>
           {loading && <AiOutlineLoading className="spinner-icon"/>} Sign In
          </Button>

          <OAuth />

          <p className="text-center mt-3">
            Don't have an account?{" "}
            <Link to="/sign-up" style={{ textDecoration: "none" }}>
              Sign Up
            </Link>
          </p>
        </Form>
      </Card.Body>
    </Card>
  </Container>
);
};

export default SignIn;
