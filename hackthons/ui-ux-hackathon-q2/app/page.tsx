import Image from "next/image";
import "./home.css"
export default function Home() {
  return (
    <div>
      <div className="hero-section flex items-center justify-center ">
        <Image src="/homepage/hero-1.png" alt="light" width={387} height={387} className="hero-pic-1"/>
        <Image src="/homepage/hero-2.png" alt="discount" width={660} height={660} className="hero-pic-2"/>
        <Image src="/homepage/hero-3.png" alt="dot" width={15} height={15} className="hero-pic-3"/>
        <div className="hero-inner-section">
        <p className="hero-p-1">Best Furniture For Your Castle....</p>
        <h1 className="hero-heading">New Furniture Collection <br/>Trends in 2020</h1>
        <p className="hero-p-2"> Lorem ipsum dolor sit amet, consectetur adipiscing elit. Magna in est adipiscing  <br/>in phasellus non in justo.</p>
        <button className="shopnowbtn">Shop Now</button>
        </div>
        <Image src="/homepage/hero-page-nav.png" alt="Pageination" width={66} height={14} className="hero-pagi"/>
      </div>

      <div className="featured-sec flex flex-col justify-center items-center">
        <h2>Featured Products</h2>
        <div className="feautred-pro grid grid-cols-4 gap-x-5">
          <div className="product">
            <div className="product-img flex items-center justify-center">
            <Image src="/homepage/featured1.png" width={175} height={175} alt="featured Chair"/>
            </div>
            <div className="product-details text-center flex flex-col gap-y-3 mt-4" >
              <h3 className="featured-pro-name">Cantilever Chair</h3>
              <div className="flex justify-center items-center gap-x-1">
              <button className="color-btn-1"></button>
              <button className="color-btn-2"></button>
              <button className="color-btn-3"></button>
              </div>
              <p className="featured-para featured-pro-code">Code - Y523201</p>
              <p className=" featured-para featured-price">$42.00</p>
            </div>
          </div>
          
          <div className="product second-prod ">
            <div className="product-img flex items-center justify-center " >
              <div className="producticons flex gap-x-3">
                <Image src="/icons/featured-cart.svg" alt="icon" width={15} height={15}/>
                <Image src="/icons/featured-heart.svg" alt="icon" width={15} height={15}/>
                <Image src="/icons/featured-search.svg" alt="icon" width={15} height={15}/>
              </div>
            <button className="featured-vm-btn">View More</button>
            <Image src="/homepage/featured2.png" width={130} height={150} alt="featured Chair"/>
            </div>
            <div className="product-details text-center flex flex-col gap-y-3 mt-4 second-prod-details" >
              <h3 className="featured-pro-name">Cantilever Chair</h3>
              <div className="flex justify-center items-center gap-x-1">
              <button className="color-btn-1"></button>
              <button className="color-btn-2"></button>
              <button className="color-btn-3 second-prod-btn"></button>
              </div>
              <p className="featured-para featured-pro-code">Code - Y523201</p>
              <p className=" featured-para featured-price">$42.00</p>
            </div>
          </div>

          <div className="product ">
            <div className="product-img flex items-center justify-center">
            <Image src="/homepage/featured3.png" width={175} height={175} alt="featured Chair"/>
            </div>
            <div className="product-details text-center flex flex-col gap-y-3 mt-4" >
              <h3 className="featured-pro-name">Cantilever Chair</h3>
              <div className="flex justify-center items-center gap-x-1">
              <button className="color-btn-1"></button>
              <button className="color-btn-2"></button>
              <button className="color-btn-3"></button>
              </div>
              <p className="featured-para featured-pro-code">Code - Y523201</p>
              <p className=" featured-para featured-price">$42.00</p>
            </div>
          </div>

          <div className="product">
            <div className="product-img flex items-center justify-center">
            <Image src="/homepage/featured4.png" width={216} height={151} alt="featured Chair"/>
            </div>
            <div className="product-details text-center flex flex-col gap-y-3 mt-4" >
              <h3 className="featured-pro-name">Cantilever Chair</h3>
              <div className="flex justify-center items-center gap-x-1">
              <button className="color-btn-1"></button>
              <button className="color-btn-2"></button>
              <button className="color-btn-3"></button>
              </div>
              <p className="featured-para featured-pro-code">Code - Y523201</p>
              <p className=" featured-para featured-price">$42.00</p>
            </div>
          </div>

        </div>
          <div className="pagi-btns flex justify-center mt-10 gap-x-2">
            <button className="btn-1 "></button>
            <button className="btn-2 pagi-btn"></button>
            <button className="btn-3 pagi-btn"></button>
            <button className="btn-4 pagi-btn"></button>
          </div>
      </div>

      <div className="latest-sec flex flex-col justify-center items-center gap-y-6">
          <h2>Latest Products</h2>
          <div className="latest-option flex justify-center gap-10">
            <p className="arrival-para">New Arrival</p>
            <p>Best Seller</p>
            <p>Featured</p>
            <p>Special Offer</p>
          </div>
          <div className="latest-prod-grid grid grid-cols-3 gap-y-5" >

            <div className="latest-prod">
              <div className="latest-prod-img flex justify-center items-center ">
                <Image src="/homepage/latest1.png" alt="latest product" width={223} height={229}/>
              </div>
              <div className="latest-prod-details flex justify-between items-center">
                <div>
                <p className="latest-prod-name">Comfort Handy Craft</p>
                <div className="underline"></div>
                </div>
                <p className="latest-prod-price">$42.00 <span className="latest-discounted">$62.00</span></p>
              </div>
            </div>

            <div className="latest-prod">
              <div className="latest-prod-img flex justify-center items-center ">
                <Image src="/homepage/latest2.png" alt="latest product" width={223} height={229}/>
              </div>
              <div className="latest-prod-details flex justify-between items-center">
                <div>
                <p className="latest-prod-name">Comfort Handy Craft</p>
                <div className="underline"></div>
                </div>
                <p className="latest-prod-price">$42.00 <span className="latest-discounted">$62.00</span></p>
              </div>
            </div>

            <div className="latest-prod">
              <div className="latest-prod-img flex justify-center items-center ">
                <Image src="/homepage/latest3.png" alt="latest product" width={223} height={229}/>
              </div>
              <div className="latest-prod-details flex justify-between items-center">
                <div>
                <p className="latest-prod-name">Comfort Handy Craft</p>
                <div className="underline"></div>
                </div>
                <p className="latest-prod-price">$42.00 <span className="latest-discounted">$62.00</span></p>
              </div>
            </div>

            <div className="latest-prod">
              <div className="latest-prod-img flex justify-center items-center ">
                <Image src="/homepage/latest4.png" alt="latest product" width={223} height={229}/>
              </div>
              <div className="latest-prod-details flex justify-between items-center">
                <div>
                <p className="latest-prod-name">Comfort Handy Craft</p>
                <div className="underline"></div>
                </div>
                <p className="latest-prod-price">$42.00 <span className="latest-discounted">$62.00</span></p>
              </div>
            </div>

            <div className="latest-prod">
              <div className="latest-prod-img flex justify-center items-center ">
                <Image src="/homepage/latest5.png" alt="latest product" width={223} height={229}/>
              </div>
              <div className="latest-prod-details flex justify-between items-center">
                <div>
                <p className="latest-prod-name">Comfort Handy Craft</p>
                <div className="underline"></div>
                </div>
                <p className="latest-prod-price">$42.00 <span className="latest-discounted">$62.00</span></p>
              </div>
            </div>

            <div className="latest-prod">
              <div className="latest-prod-img flex justify-center items-center ">
                <Image src="/homepage/latest6.png" alt="latest product" width={223} height={229}/>
              </div>
              <div className="latest-prod-details flex justify-between items-center">
                <div>
                <p className="latest-prod-name">Comfort Handy Craft</p>
                <div className="underline"></div>
                </div>
                <p className="latest-prod-price">$42.00 <span className="latest-discounted">$62.00</span></p>
              </div>
            </div>
          </div>
      </div>

      <div className="offer-sec flex flex-col justify-center items-center">
        <h2>What Shopex Offer!</h2>
        <div className="offer-cards-sec grid grid-cols-4 ">
          <div className="offer-card flex flex-col justify-center items-center text-center gap-y-5">
            <Image src="/homepage/free-delivery.png" alt="free delivery" width={65} height={65}/>
            <p className="offer-card-p">24/7 Support</p>
            <p className="offer-card-descrip">Lorem ipsum dolor sit amet, consectetur adipiscing elit. <br />Massa purus gravida.</p>
          </div>

          <div className="offer-card flex flex-col justify-center items-center text-center gap-y-5">
            <Image src="/homepage/cashback.png" alt="free delivery" width={65} height={65}/>
            <p className="offer-card-p">24/7 Support</p>
            <p className="offer-card-descrip">Lorem ipsum dolor sit amet, consectetur adipiscing elit. <br />Massa purus gravida.</p>
          </div>

          <div className="offer-card flex flex-col justify-center items-center text-center gap-y-5">
            <Image src="/homepage/premium-quality.png" alt="free delivery" width={65} height={65}/>
            <p className="offer-card-p">24/7 Support</p>
            <p className="offer-card-descrip">Lorem ipsum dolor sit amet, consectetur adipiscing elit. <br />Massa purus gravida.</p>
          </div>

          <div className="offer-card flex flex-col justify-center items-center text-center gap-y-5">
            <Image src="/homepage/24-hours-support.png" alt="free delivery" width={65} height={65}/>
            <p className="offer-card-p">24/7 Support</p>
            <p className="offer-card-descrip">Lorem ipsum dolor sit amet, consectetur adipiscing elit. <br />Massa purus gravida.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
