import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Easy to Learn',
    description: (
      <>
        Comprehensive curriculum designed for beginners to learn Physical AI
        and Humanoid Robotics step by step.
      </>
    ),
  },
  {
    title: 'Focus on Practice',
    description: (
      <>
        Hands-on exercises and practical examples to reinforce theoretical concepts.
      </>
    ),
  },
  {
    title: 'Real-World Ready',
    description: (
      <>
        Learn deployment strategies and safety considerations for real-world applications.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}