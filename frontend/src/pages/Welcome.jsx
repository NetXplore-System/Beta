import React, { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
  motion,
  useScroll,
  useTransform,
  AnimatePresence,
  useAnimation,
  useInView,
} from "framer-motion";
import { Button } from "react-bootstrap";
import {
  FaSearch,
  FaNetworkWired,
  FaUsers,
  FaChartLine,
  FaArrowDown,
  FaFilter,
  FaWhatsapp,
  FaWikipediaW,
  FaFileUpload,
  FaRocket,
} from "react-icons/fa";
import NetworkAnimation from "../components/NetworkAnimation";
import NetworkMockupVisualization from "../components/NetworkMockupVisualization";
import VideoPopup from "../components/utils/VideoPopup";
import "../styles/Welcome.css";

const sections = [
  {
    title: "Upload & Analyze Any Conversation",
    text: "Instantly transform your WhatsApp chats or Wikipedia data into interactive network visualizations. Upload your data with a simple click and watch as connections form before your eyes, revealing the hidden patterns within your conversations.",
    backgroundColor: "linear-gradient(135deg, #050d2d 0%, #0a1b3d 100%)",
    icon: <FaFileUpload size={40} />,
    imageSrc: "/images/data-upload.webp",
    features: [
      "WhatsApp chat export compatibility",
      "Wikipedia data integration",
      "Secure data handling",
      "Instant visualization",
    ],
  },
  {
    title: "Discover Hidden Network Patterns",
    text: "Uncover the invisible structures within conversations using our powerful network analysis tools. Filter by dates, keywords, users, or message length to identify communication patterns and key influencers within any discourse.",
    backgroundColor: "linear-gradient(135deg, #0a1b3d 0%, #1c1c2e 100%)",
    icon: <FaSearch size={40} />,
    imageSrc: "/images/pattern-discovery.webp",
    features: [
      "Advanced message filtering",
      "Temporal analysis",
      "Keyword tracking",
      "User activity metrics",
    ],
  },
  {
    title: "Identify Communities & Key Influencers",
    text: "Advanced algorithms automatically detect communities within your network and highlight the most influential participants. Calculate PageRank, closeness, and betweenness centrality metrics to understand who drives the conversation.",
    backgroundColor: "linear-gradient(135deg, #1c1c2e 0%, #0c3944 100%)",
    icon: <FaUsers size={40} />,
    imageSrc: "/images/community-detection.webp",
    features: [
      "Community detection algorithms",
      "Centrality metrics",
      "Influence ranking",
      "Group dynamics analysis",
    ],
  },
  {
    title: "Compare Networks & Test Scenarios",
    text: "Compare different conversation networks, simulate the removal of users, or hide inter-community links to understand resilience and information flow. Analyze network density and run diameter to understand local structures.",
    backgroundColor: "linear-gradient(135deg, #0c3944 0%, #11224d 100%)",
    icon: <FaChartLine size={40} />,
    imageSrc: "/images/network-comparison.webp",
    features: [
      "Network comparison tools",
      "What-if scenario testing",
      "Structure resilience testing",
    ],
  },
];

const FeatureCard = ({ icon, title, description }) => (
  <div className="feature-card">
    <div className="feature-card-icon">{icon}</div>
    <h3 className="feature-card-title">{title}</h3>
    <p className="feature-card-description">{description}</p>
  </div>
);

const AnimatedButton = ({ variant, onClick, children }) => {
  return (
    <motion.div
      whileHover={{
        scale: 1.05,
        boxShadow: "0 8px 20px rgba(13, 110, 253, 0.5)",
      }}
      whileTap={{ scale: 0.95 }}
      className="animated-button-wrapper"
    >
      <Button
        variant={variant}
        size="lg"
        onClick={onClick}
        className={`animated-button ${
          variant === "primary" ? "primary-button" : "outline-button"
        }`}
      >
        {children}
      </Button>
    </motion.div>
  );
};

const FloatingElement = ({
  children,
  delay,
  duration,
  yRange,
  xRange,
  className,
}) => {
  return (
    <motion.div
      animate={{
        y: yRange,
        x: xRange,
      }}
      transition={{
        repeat: Infinity,
        repeatType: "reverse",
        duration: duration,
        delay: delay,
        ease: "easeInOut",
      }}
      className={`floating-element ${className || ""}`}
    >
      {children}
    </motion.div>
  );
};

const AnimatedTextReveal = ({ text, delay = 0 }) => {
  return (
    <span className="text-reveal-wrapper">
      <motion.span
        initial={{ y: "100%" }}
        animate={{ y: 0 }}
        transition={{ duration: 0.8, delay: delay, ease: "easeOut" }}
        className="text-reveal"
      >
        {text}
      </motion.span>
    </span>
  );
};

const FeatureBadge = ({ text }) => {
  return (
    <motion.span whileHover={{ scale: 1.05, y: -3 }} className="feature-badge">
      {text}
    </motion.span>
  );
};

const FeatureSection = ({ section, index }) => {
  const controls = useAnimation();
  const ref = useRef(null);
  const inView = useInView(ref, { threshold: 0.15, triggerOnce: true });
  const { scrollYProgress } = useScroll({ target: ref });

  const parallaxY = useTransform(scrollYProgress, [0, 1], [50, -50]);
  const parallaxOpacity = useTransform(
    scrollYProgress,
    [0, 0.3, 0.8],
    [0.3, 1, 0.8]
  );

  useEffect(() => {
    if (inView) {
      controls.start("visible");
    }
  }, [controls, inView]);

  const isEven = index % 2 === 0;

  const fadeInVariant = {
    hidden: { opacity: 0, y: 70 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.9, ease: "easeOut", staggerChildren: 0.2 },
    },
  };

  const imageVariant = {
    hidden: { opacity: 0, x: isEven ? -100 : 100, scale: 0.9 },
    visible: {
      opacity: 1,
      x: 0,
      scale: 1,
      transition: { duration: 1.2, ease: "easeOut", delay: 0.3 },
    },
  };

  return (
    <section
      className="feature-section"
      ref={ref}
      style={{ background: section.backgroundColor }}
    >
      <motion.div
        className="gradient-circle top"
        style={{
          left: isEven ? "10%" : "auto",
          right: isEven ? "auto" : "10%",
          y: parallaxY,
          opacity: parallaxOpacity,
        }}
      />

      <motion.div
        className="gradient-circle bottom"
        style={{
          right: isEven ? "15%" : "auto",
          left: isEven ? "auto" : "15%",
        }}
      />

      <div className="container">
        <div
          className="row align-items-center"
          style={{ flexDirection: isEven ? "row" : "row-reverse" }}
        >
          <motion.div
            className="col-lg-6 mb-5 mb-lg-0"
            initial="hidden"
            animate={controls}
            variants={fadeInVariant}
          >
            <motion.div
              variants={fadeInVariant}
              className="feature-icon-container"
            >
              <motion.div
                whileHover={{ rotate: 360 }}
                transition={{ duration: 1.5, ease: "easeInOut" }}
                className="feature-icon"
              >
                {section.icon}
              </motion.div>
            </motion.div>

            <motion.h2 variants={fadeInVariant} className="feature-title">
              {section.title}
              <span className="feature-title-underline" />
            </motion.h2>

            <motion.p variants={fadeInVariant} className="feature-text">
              {section.text}
            </motion.p>

            <motion.div variants={fadeInVariant} className="feature-badges">
              {section.features.map((feature, i) => (
                <FeatureBadge key={i} text={feature} />
              ))}
            </motion.div>
          </motion.div>

          <motion.div
            className="col-lg-6"
            initial="hidden"
            animate={controls}
            variants={imageVariant}
          >
            <NetworkMockupVisualization />
          </motion.div>
        </div>
      </div>
    </section>
  );
};

const DataSourceBadge = ({ icon, text }) => {
  return (
    <motion.div whileHover={{ y: -5, scale: 1.05 }} className="data-badge">
      <div className="data-badge-icon">{icon}</div>
      <div className="data-badge-text">{text}</div>
    </motion.div>
  );
};

const Welcome = () => {
  const navigate = useNavigate();
  const [showVideo, setShowVideo] = useState(false);
  const [scrollY, setScrollY] = useState(0);
  const titleRef = useRef(null);
  const subtitleRef = useRef(null);
  const inViewTitle = useInView(titleRef, { once: true, amount: 0.5 });
  const inViewSubtitle = useInView(subtitleRef, { once: true, amount: 0.5 });

  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  useEffect(() => {
    document.body.style.overflow = "auto";
    document.body.style.margin = "0";
    document.body.style.backgroundColor = "#050d2d";
    document.body.style.fontFamily = "'Inter', sans-serif";

    return () => {
      document.body.style.overflow = "";
      document.body.style.margin = "";
      document.body.style.backgroundColor = "";
      document.body.style.fontFamily = "";
    };
  }, []);

  const heroFadeInUpStrong = {
    hidden: { opacity: 0, y: 100 },
    visible: (i) => ({
      opacity: 1,
      y: 0,
      transition: {
        duration: 1,
        ease: "easeOut",
        delay: i * 0.2,
      },
    }),
  };

  const ctaFadeIn = {
    hidden: { opacity: 0, scale: 0.9 },
    visible: {
      opacity: 1,
      scale: 1,
      transition: {
        duration: 0.8,
        ease: "easeOut",
        delay: 0.6,
      },
    },
  };

  return (
    <div className="app-wrapper">
      <section className="hero-section">
        <NetworkAnimation />

        <div className="hero-overlay"></div>

        <div className="hero-content">
          <FloatingElement
            delay={0}
            duration={6}
            yRange={["30px", "-30px"]}
            xRange={["10px", "-10px"]}
            className="floating-element-1"
          />

          <FloatingElement
            delay={1}
            duration={7}
            yRange={["-20px", "20px"]}
            xRange={["-15px", "15px"]}
            className="floating-element-2"
          />

          <FloatingElement
            delay={0.5}
            duration={5}
            yRange={["15px", "-15px"]}
            xRange={["5px", "-5px"]}
            className="floating-element-3"
          />

          <div ref={titleRef} className="title-wrapper">
            <motion.h1
              custom={0}
              initial="hidden"
              animate={inViewTitle ? "visible" : "hidden"}
              variants={heroFadeInUpStrong}
              className="hero-title"
            >
              <div className="title-first-line">
                <AnimatedTextReveal text="Unlock the Power of" delay={0.1} />
              </div>
              <div className="hero-accent">
                <AnimatedTextReveal text="Network Analysis" delay={0.5} />
                <motion.span
                  initial={{ width: 0 }}
                  animate={{ width: "100%" }}
                  transition={{ delay: 1.2, duration: 1, ease: "easeOut" }}
                  className="accent-underline"
                ></motion.span>
              </div>
            </motion.h1>
          </div>

          <motion.div
            ref={subtitleRef}
            custom={1}
            initial="hidden"
            animate={inViewSubtitle ? "visible" : "hidden"}
            variants={heroFadeInUpStrong}
            className="hero-subtitle-container"
          >
            <p className="hero-subtitle">
              Transform conversations into interactive networks. Identify key
              influencers, discover hidden communities, and gain powerful
              insights from your social data.
            </p>

            <div className="data-sources">
              <DataSourceBadge
                icon={<FaWhatsapp size={24} color="#25D366" />}
                text="WhatsApp Data"
              />
              <DataSourceBadge
                icon={<FaWikipediaW size={24} color="#ffffff" />}
                text="Wikipedia Data"
              />
            </div>
          </motion.div>

          <div className="hero-buttons">
            <motion.div
              initial={{ opacity: 0, scale: 0.8, filter: "blur(8px)" }}
              animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
              transition={{ duration: 1.2, ease: "easeOut", delay: 1.5 }}
            >
              <AnimatedButton
                variant="primary"
                onClick={() => navigate("/choose-platform")}
              >
                Start Analyzing
              </AnimatedButton>
              <AnimatedButton
                variant="outline-light"
                onClick={() => navigate("/signin")}
              >
                Log In
              </AnimatedButton>
            </motion.div>
          </div>
        </div>
      </section>

      {sections.map((section, index) => (
        <FeatureSection key={index} section={section} index={index} />
      ))}

      <section className="cta-section">
        <div className="cta-overlay-1"></div>
        <div className="cta-overlay-2"></div>

        <div className="container">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.3 }}
            variants={{
              hidden: { opacity: 0, y: 50 },
              visible: {
                opacity: 1,
                y: 0,
                transition: {
                  duration: 0.8,
                  ease: "easeOut",
                  staggerChildren: 0.2,
                },
              },
            }}
            className="cta-content"
          >
            <motion.h2
              variants={{
                hidden: { opacity: 0, y: 50 },
                visible: {
                  opacity: 1,
                  y: 0,
                  transition: { duration: 0.8 },
                },
              }}
              className="cta-title"
            >
              Reveal the Hidden Structure of Communication
              <motion.span
                initial={{ width: 0 }}
                whileInView={{ width: "60%" }}
                transition={{ delay: 0.5, duration: 1 }}
                viewport={{ once: true }}
                className="cta-title-underline"
              />
            </motion.h2>

            <motion.p
              variants={{
                hidden: { opacity: 0, y: 50 },
                visible: {
                  opacity: 1,
                  y: 0,
                  transition: { duration: 0.8, delay: 0.1 },
                },
              }}
              className="cta-text"
            >
              Turn conversations into meaningful insights with advanced network
              analysis tools. Visualize connections, detect communities, and
              explore influence in social or discursive networks — all in one
              intuitive platform.
            </motion.p>

            <motion.div
              variants={{
                hidden: { opacity: 0, scale: 0.9 },
                visible: {
                  opacity: 1,
                  scale: 1,
                  transition: { duration: 0.5, delay: 0.3 },
                },
              }}
              className="cta-button-container"
            >
              <AnimatedButton
                variant="primary"
                onClick={() => setShowVideo(true)}
              >
                Watch Demo
              </AnimatedButton>
            </motion.div>

            <motion.div
              variants={{
                hidden: { opacity: 0, y: 30 },
                visible: {
                  opacity: 1,
                  y: 0,
                  transition: { duration: 0.8, delay: 0.5 },
                },
              }}
              className="feature-grid"
            >
              <FeatureCard
                icon={<FaFileUpload size={28} />}
                title="Easy Data Upload"
                description="Quickly upload WhatsApp or Wikipedia data and start exploring your network in seconds."
              />

              <FeatureCard
                icon={<FaUsers size={28} />}
                title="Community Detection"
                description="Automatically detect groups and communities in your data to uncover hidden structures."
              />

              <FeatureCard
                icon={<FaChartLine size={28} />}
                title="Network Metrics"
                description="Measure PageRank, centrality, and density to evaluate the influence and resilience of your network."
              />

              <FeatureCard
                icon={<FaSearch size={28} />}
                title="Custom Filters"
                description="Filter conversations by user, time, keywords, or message length to focus your analysis."
              />
            </motion.div>
          </motion.div>
        </div>
      </section>

      <VideoPopup
        show={showVideo}
        onHide={() => setShowVideo(false)}
        videoUrl="https://www.youtube.com/embed/cnRZiUUJspU"
      />

      <footer className="footer">
        <div className="container">
          <div className="row">
            <div className="col-md-6 mb-4 mb-md-0">
              <h3 className="footer-logo">NetXplore</h3>
              <p className="footer-text">
                Advanced network analysis tools for researchers exploring
                conversation data and social interactions.
              </p>
            </div>
            <div className="col-md-6 text-md-end">
              <div className="footer-buttons">
                <AnimatedButton
                  variant="outline-light"
                  onClick={() => navigate("/register")}
                >
                  Create Account
                </AnimatedButton>
                <AnimatedButton
                  variant="primary"
                  onClick={() => navigate("/choose-platform")}
                >
                  Start Analyzing
                </AnimatedButton>
              </div>
              <p className="footer-copyright">
                © {new Date().getFullYear()} NetXplore. All rights reserved.
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Welcome;
