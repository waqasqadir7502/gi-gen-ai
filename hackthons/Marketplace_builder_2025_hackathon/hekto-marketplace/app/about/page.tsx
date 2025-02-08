import Image from "next/image";
import "./about.css";
import PageName from "../components/Page name bar/pageName";

export default function AboutUs() {
  return (
    <div>
      <PageName name="About Us" />

      <div className="about-sec flex justify-center items-center">
        <div className="about-inner grid grid-cols-2 gap-4">
          <div className="about-col-1">
            <Image src="/about/about.png" alt="Two Man In a Meeting" width={570} height={409} />
          </div>
          <div className="about-col-2 flex flex-col justify-evenly">
            <div >
            <h3>Know About Our Ecomerce Business, History</h3>
            <p>
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. Mattis
              neque ultrices mattis aliquam, malesuada diam est. Malesuada sem
              tristique amet erat vitae eget dolor lobortis. Accumsan faucibus
              vitae lobortis quis bibendum quam.
            </p>
            </div>
            <div>
            <button>Contact us</button>
            </div>
          </div>
        </div>
      </div>

      {/* Offer Section / Section 4 */}
      <div className="offer-sec flex flex-col justify-center items-center">
        <h2>Our Features</h2>
        <div className="offer-cards-sec grid grid-cols-4 gap-x-8">
          <div className="offer-card flex flex-col justify-center items-center text-center gap-y-5">
            <Image
              src="/homepage/free-delivery.png"
              alt="free delivery"
              width={65}
              height={65}
            />
            <p className="offer-card-p">24/7 Support</p>
            <p className="offer-card-descrip">
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. <br />
              Massa purus gravida.
            </p>
          </div>

          <div className="offer-card flex flex-col justify-center items-center text-center gap-y-5">
            <Image
              src="/homepage/cashback.png"
              alt="free delivery"
              width={65}
              height={65}
            />
            <p className="offer-card-p">24/7 Support</p>
            <p className="offer-card-descrip">
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. <br />
              Massa purus gravida.
            </p>
          </div>

          <div className="offer-card flex flex-col justify-center items-center text-center gap-y-5">
            <Image
              src="/homepage/premium-quality.png"
              alt="free delivery"
              width={65}
              height={65}
            />
            <p className="offer-card-p">24/7 Support</p>
            <p className="offer-card-descrip">
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. <br />
              Massa purus gravida.
            </p>
          </div>

          <div className="offer-card flex flex-col justify-center items-center text-center gap-y-5">
            <Image
              src="/homepage/24-hours-support.png"
              alt="free delivery"
              width={65}
              height={65}
            />
            <p className="offer-card-p">24/7 Support</p>
            <p className="offer-card-descrip">
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. <br />
              Massa purus gravida.
            </p>
          </div>
        </div>
      </div>

      {/* Client Section */}
      <div className="client-sec flex justify-center items-center">
        <div className="client-inner flex flex-col items-center">
        <h3>Our Client Say!</h3>
        <Image src="/about/client.png" alt="clientle" width={689} height={285} />
        </div>
      </div>
    </div>
  );
}
