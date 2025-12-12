// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Module 1: Foundations of Physical AI',
      items: [
        'modules/module-1/intro',
        'modules/module-1/chapters/chapter-1.1-introduction-to-physical-ai',
        'modules/module-1/chapters/chapter-1.2-history-evolution-physical-ai',
        'modules/module-1/chapters/chapter-1.3-basic-mathematics-physical-ai',
        'modules/module-1/chapters/chapter-1.4-simulation-environment-setup'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Humanoid Robotics Fundamentals',
      items: [
        'modules/module-2/intro',
        'modules/module-2/chapters/chapter-2.1-kinematics-movement-systems',
        'modules/module-2/chapters/chapter-2.2-sensors-perception-systems',
        'modules/module-2/chapters/chapter-2.3-actuators-control-systems',
        'modules/module-2/chapters/chapter-2.4-basic-locomotion-patterns'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: Control and Intelligence',
      items: [
        'modules/module-3/intro',
        'modules/module-3/chapters/chapter-3.1-balance-stability-control',
        'modules/module-3/chapters/chapter-3.2-path-planning-navigation',
        'modules/module-3/chapters/chapter-3.3-manipulation-grasping',
        'modules/module-3/chapters/chapter-3.4-learning-based-control'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Applications and Integration',
      items: [
        'modules/module-4/intro',
        'modules/module-4/chapters/chapter-4.1-human-robot-interaction',
        'modules/module-4/chapters/chapter-4.2-multi-sensor-fusion',
        'modules/module-4/chapters/chapter-4.3-real-world-deployment',
        'modules/module-4/chapters/chapter-4.4-capstone-project'
      ],
    },
  ],
};

module.exports = sidebars;