import React, { useState } from "react";
import { Link } from "react-router-dom";
import Button from "react-bootstrap/Button";
import Form from "react-bootstrap/Form";
import Container from "react-bootstrap/Container";
import OAuth from '../components/OAuth.jsx';
import { useNavigate } from "react-router-dom";
import Card from "react-bootstrap/Card";
import toast from "react-hot-toast";
import { useDispatch } from "react-redux";
import { setUser } from "../redux/user/userSlice.js";
import { AiOutlineLoading } from "react-icons/ai";


const SignUp = () => {
  const dispatch = useDispatch();
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  // const [success, setSuccess] = useState("");
  const navigate = useNavigate();
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
    // setSuccess(""); 

    const { name, email, password, confirmPassword } = formData;

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    try {
      setLoading(true);
      const response = await fetch(import.meta.env.VITE_API_URL + "/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ name, email, password }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Something went wrong");
      }

      const data = await response.json();
      toast.success("Registration successful!");
      dispatch(setUser(data));
      navigate("/choose-platform");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container
      className="d-flex justify-content-center align-items-center pb-5"
      style={{ minHeight: "100vh" }}
    >
      <Card style={{ width: "100%", maxWidth: "500px", borderRadius: "20px" }} className="shadow">
        <Card.Body className="p-4">
          <h3 className="text-center mb-4">Sign Up</h3>

          {error && <p className="text-danger text-center">{error}</p>}
          {/* {success && <p className="text-success text-center">{success}</p>} */}

          <Form onSubmit={handleSubmit}>
            <Form.Group className="mb-3" controlId="formBasicName">
              <Form.Label>Full Name</Form.Label>
              <Form.Control
                type="text"
                name="name"
                value={formData.name}
                onChange={handleChange}
                placeholder="Enter your full name"
                required
              />
            </Form.Group>

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

            <Form.Group className="mb-3" controlId="formBasicConfirmPassword">
              <Form.Label>Confirm Password</Form.Label>
              <Form.Control
                type="password"
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleChange}
                placeholder="Confirm Password"
                required
              />
            </Form.Group>

            <Button
              type="submit"
              className="w-100 mb-3"
              disabled={loading}
              style={{ backgroundColor: "#050d2d", borderColor: "#050d2d" }}
            >
              {loading && <AiOutlineLoading className="spinner-icon" />} Sign Up
            </Button>

            <OAuth />

            <p className="text-center mt-3">
              Already have an account?{" "}
              <Link to="/sign-in" style={{ textDecoration: "none" }}>
                Sign In
              </Link>
            </p>
          </Form>
        </Card.Body>
      </Card>
    </Container>
  );
};

export default SignUp;
