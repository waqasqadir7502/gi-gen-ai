import Image from "next/image";
import Link from "next/link";
import "./style/style.css";

export default function Home() {
  return (
    <div>
      <nav>
        <ul>
          <li>
            <Link href="#">Work</Link>
          </li>
          <li>
            <Link href="#">Blog</Link>
          </li>
          <li>
            <Link href="#">Contact</Link>
          </li>
        </ul>
      </nav>

      <div className="hero-sec">
        <div className="hero-sec-1">
          <h1>
            Hi, I am John, <br /> Creative Technologist
          </h1>
          <p>
            Amet minim mollit non deserunt ullamco est sit aliqua dolor do amet
            sint. Velit officia consequat duis enim velit mollit. Exercitation
            veniam consequat sunt nostrud amet.
          </p>
          <button >Download Resume</button>
        </div>
        <div className="hero-sec-2">
          <Image
            src="/hero-image.png"
            width={298}
            height={308}
            alt="hero-img"
          />
        </div>
      </div>

      <div className="recent">
        <div className="recent-container">
          <div className="recent-para">
            <p>Recent Post</p>
            <Link href="#">View More</Link>
          </div>

          <div className="posts">
            <div className="post-1">
              <div className="post-1-inner">
                <h2>Making a design system from scratch</h2>
                <div className="mid-sec">
                  <p>12 Feb 2020</p>
                  <span className="vector"></span>
                  <p>Design, Pattern</p>
                </div>
                <p>
                  Amet minim mollit non deserunt ullamco est sit aliqua dolor do
                  amet sint. Velit officia consequat duis enim velit mollit.
                  Exercitation veniam consequat sunt nostrud amet.
                </p>
              </div>
            </div>

            <div className="post-1">
            <div className="post-1-inner">
            <h2>Making a design system from scratch</h2>
            <div className="mid-sec">
              <p>12 Feb 2020</p>
              <span className="vector"></span>
              <p>Design, Pattern</p>
            </div>
            <p>
              Amet minim mollit non deserunt ullamco est sit aliqua dolor do
              amet sint. Velit officia consequat duis enim velit mollit.
              Exercitation veniam consequat sunt nostrud amet.
            </p>
            </div>
          </div>
          </div>
        </div>
      </div>

      <div className="featured">
        <h2>Featured works</h2>
        <div className="ft-1">
          <div className="ft-sec-1">
            <Image src="/Rectangle 30.png" alt="featured pic 1" width={246} height={180} />
          </div>
          <div className="ft-sec-2">
            <h2>Designing Dashboards</h2>
            <div className="ft-mid-sec">
              <span>2020</span>
              <p>Dashboard</p>
            </div>
            <p>
              Amet minim mollit non deserunt ullamco est sit aliqua dolor do
              amet sint. Velit officia consequat duis enim velit mollit.
              Exercitation veniam consequat sunt nostrud amet.
            </p>
          </div>
        </div>

        <div className="ft-1">
          <div className="ft-sec-1">
            <Image src="/Rectangle 32.png" alt="featured pic 2" width={246} height={180} />
          </div>
          <div className="ft-sec-2">
            <h2>Designing Dashboards</h2>
            <div className="ft-mid-sec">
              <span>2020</span>
              <p>Dashboard</p>
            </div>
            <p>
              Amet minim mollit non deserunt ullamco est sit aliqua dolor do
              amet sint. Velit officia consequat duis enim velit mollit.
              Exercitation veniam consequat sunt nostrud amet.
            </p>
          </div>
        </div>

        <div className="ft-1">
          <div className="ft-sec-1">
            <Image src="/Rectangle 34.png" alt="featured pic 3" width={246} height={180} />
          </div>
          <div className="ft-sec-2">
            <h2>Designing Dashboards</h2>
            <div className="ft-mid-sec">
              <span>2020</span>
              <p>Dashboard</p>
            </div>
            <p>
              Amet minim mollit non deserunt ullamco est sit aliqua dolor do
              amet sint. Velit officia consequat duis enim velit mollit.
              Exercitation veniam consequat sunt nostrud amet.
            </p>
          </div>
        </div>

      </div>

      <footer>
        <div className="icons">
       <Image src="/fb.svg" alt="facebook" width={30} height={30}/>
       <Image src="/insta.svg" alt="instragram"  width={30} height={30}/>
       <Image src="/Vector.svg" alt="twitter"  width={30} height={30}/>
       <Image src="/Linkedin.svg" alt="Linkedin"  width={30} height={30}/>
        </div> 
        <p>Copyright ©2020 All rights reserved</p>
      </footer>
    </div>
  );
}
