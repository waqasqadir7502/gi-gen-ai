import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/modules/module-1/intro">
            Start Learning - 5min ⏱️
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="A comprehensive educational resource for learning Physical AI and Humanoid Robotics">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              <div className="col col--4">
                <div className="text--center padding-horiz--md">
                  <h3>Physical AI Foundations</h3>
                  <p>Learn the fundamental concepts of Physical AI, from basic mathematics to simulation environments.</p>
                </div>
              </div>
              <div className="col col--4">
                <div className="text--center padding-horiz--md">
                  <h3>Humanoid Robotics</h3>
                  <p>Explore the mechanics of humanoid robots, including kinematics, sensors, actuators, and locomotion.</p>
                </div>
              </div>
              <div className="col col--4">
                <div className="text--center padding-horiz--md">
                  <h3>Real-World Applications</h3>
                  <p>Discover how to deploy humanoid robots in real-world scenarios with safety and ethical considerations.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.modules}>
          <div className="container padding-vert--lg">
            <div className="row">
              <div className="col col--12">
                <h2 className="text--center">Complete Learning Path</h2>
                <p className="text--center">
                  Four comprehensive modules that take you from beginner to advanced concepts in Physical AI and Humanoid Robotics.
                </p>
              </div>
            </div>

            <div className="row padding-vert--lg">
              <div className="col col--3">
                <div className="text--center">
                  <h3>Module 1</h3>
                  <p>Foundations of Physical AI</p>
                </div>
              </div>
              <div className="col col--3">
                <div className="text--center">
                  <h3>Module 2</h3>
                  <p>Humanoid Robotics Fundamentals</p>
                </div>
              </div>
              <div className="col col--3">
                <div className="text--center">
                  <h3>Module 3</h3>
                  <p>Control and Intelligence</p>
                </div>
              </div>
              <div className="col col--3">
                <div className="text--center">
                  <h3>Module 4</h3>
                  <p>Applications and Integration</p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}